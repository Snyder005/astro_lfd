__all__ = ["ADRTDetectConfig", "ADRTDetectTask", "AdrtPeak"]

from dataclasses import dataclass

import adrt
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .base import get_line_mask, get_pixel_mask, timed
from ..geom.line import Line2D
from ..table.streakAdapter import StreakAdapter


class ADRTDetectConfig(pexConfig.Config):
    """Configurable parameters for `ADRTDetectTask`."""

    # Configuration for preprocess
    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    bad_mask_dilation = pexConfig.Field(
        doc="Number of pixels to dilate the bad mask by.",
        dtype=int,
        default=1,
    )
    stats_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when calculating statistics.",
        dtype=str,
        default=["BAD", "SAT", "EDGE", "NO_DATA"],
    )

    # Configuration for postprocess
    detected_mask_plane = pexConfig.Field(
        doc="Name of mask plane with pixels above detection threshold.",
        dtype=str,
        default="DETECTED",
    )
    only_mask_detected = pexConfig.Field(
        doc="If true, only propagate the part of the streak mask that overlaps with the detection mask.",
        dtype=bool,
        default=True,
    )
    streak_width = pexConfig.Field(
        doc="Initial width (in pixels) of the streak mask.",
        dtype=float,
        default=200.0,
    )


class ADRTDetectTask(pipeBase.Task):
    """Detect linear features with the Approximate Discrete Radon Transform."""

    ConfigClass = ADRTDetectConfig
    _DefaultName = "adrtDetect"

    timings: dict[str, float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = {}

    def run(self, table: afwTable.SourceTable, exposure: afwImage.ExposureF) -> pipeBase.Struct:
        """Detect streaks in an exposure.

        Parameters
        ----------
        table : `lsst.afw.table.SourceTable`
            The source table used to construct the output catalog. Its schema
            must provide the streak ``line_*`` fields (see
            `~astro_lfd.table.streakAdapter.StreakAdapter.makeMinimalSchema`).
        exposure : `lsst.afw.image.ExposureF`
            The exposure to search. The mask plane named by
            ``config.detected_mask_plane`` must flag the detected pixels.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            The task result as a struct with attributes:

            ``streaks``
                Catalog of detected streaks (`lsst.afw.table.SourceCatalog`).
            ``imarr``
                Image array with invalid regions masked (`numpy.ndarray`).
            ``streak_mask``
                Streak mask plane (`numpy.ndarray`).
            ``timings``
                Computing times for each processing step (`dict`)
        """
        streaks = afwTable.SourceCatalog(table)
        self.timings = {}

        imarr, bad_mask, sigma  = self.preprocess(exposure)
        lines = self.detect(imarr, bad_mask, sigma)
        streak_mask = self.postprocess(streaks, exposure, lines)

        return pipeBase.Struct(
            streaks=streaks,
            imarr=imarr,
            bad_mask=bad_mask,
            streak_mask=streak_mask,
            timings=self.timings,
        )

    @timed("preprocess")
    def preprocess(self, exposure: afwImage.ExposureF) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Perform preprocessing on input exposure.

        Parameters
        ----------
        exposure: `lsst.afw.image.ExposureF`
            The exposure to search.

        Returns
        -------
        imarr : `numpy.ndarray`
            The image array with invalid regions masked.
        bad_mask : `numpy.ndarray`
            The bad pixel mask array
        """
        bad_mask = get_pixel_mask(
            exposure.mask,
            self.config.bad_mask_planes,
            dilation=self.config.bad_mask_dilation,
        )
        padded_mask = np.ones((4096, 4096), dtype=bool)
        padded_mask[:bad_mask.shape[0], :bad_mask.shape[1]] = bad_mask

        imarr = exposure.image.array.copy()
        padded_imarr = np.zeros((4096, 4096))
        padded_imarr[:imarr.shape[0], :imarr.shape[1]] = imarr
        padded_imarr[padded_mask] = 0.0

        bad = exposure.mask.getPlaneBitMask(self.config.stats_mask_planes)
        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(bad)
        stats = afwMath.makeStatistics(exposure.maskedImage, afwMath.STDEVCLIP, sctrl)
        sigma = stats.getValue(afwMath.STDEVCLIP)

        return padded_imarr, padded_mask, sigma

    @timed("detect")
    def detect(
        self,
        imarr: NDArray[np.float64],
        bad_mask: NDArray[np.bool_],
        sigma: float,
        footprint: tuple[int, int] = (7, 31),
        contrast_annulus: tuple[int, int] = (12, 40),
        abs_min_threshold: float = 20.0,  # Elevate to configurable parameter
        min_length: float = 64.0,
        max_peaks: int = 50,
        cone_pad: int = 4,
        wing_margin: float = 1.0,
        wing_threshold = 30.0,  # Elevate to configurable parameter
        dedup_rho: float = 3.0,
        dedup_theta: float = 0.5
    ) -> list[Line2D]:
        """Perform linear feature detection on masked image array.

        Runs the forward ADRT and the multi-peak detector `_find_peaks` on the
        accumulator. Lines are returned in the PIXEL frame of the grid the
        ADRT ran on, ranked by descending significance.

        Parameters
        ----------
        imarr : `numpy.ndarray`
            The masked image array (square, power-of-two size).

        Returns
        -------
        lines : `list` [`~astro_lfd.geom.Line2D`]
            The detected lines, in the PIXEL frame.
        """
        L = adrt.adrt((~bad_mask).astype(float))
        S = adrt.adrt(imarr) / (sigma * np.sqrt(np.maximum(L, 1.0)))
        N = 4096

        # Stage 1: local maxima above a robust threshold.
        is_max = S == ndimage.maximum_filter(S, size=(1, *footprint), mode="nearest")
        qhs = np.argwhere(is_max & (S > abs_min_threshold) & (L >= min_length))
        n_local = len(qhs)

        # Stage 2: height-axis local contrast. Wings are nearly as high
        # tens of cells away along h; a real line peak is not.
        if len(qhs):
            off = np.arange(*contrast_annulus)
            off = np.concatenate([-off, off])
            hh = np.clip(qhs[:, 1, None] + off[None, :], 0, 2 * N - 2)
            background = np.median(S[qhs[:, 0, None], hh, qhs[:, 2, None]], axis=1)
            values = S[qhs[:, 0], qhs[:, 1], qhs[:, 2]]
            keep = (values - background) > abs_min_threshold
            qhs = qhs[keep]
            values = values[keep]
            order = np.argsort(-values, kind="stable")
            qhs = qhs[order]
            values = values[order]
        else:
            values = np.empty(0, dtype=np.float32)
        n_contrast = len(qhs)

        # Stage 3: greedy cone suppression. Every candidate is re-expressed in
        # the accepted peak's quadrant frame so wings that cross a seam are
        # still inside its cone.
        rho_all, theta_all = _adrt_to_hesse(qhs[:, 0], qhs[:, 1], qhs[:, 2], N)
        length_all = L[qhs[:, 0], qhs[:, 1], qhs[:, 2]]
        alive = np.ones(len(qhs), dtype=bool)
        accepted: list[int] = []
        while alive.any() and len(accepted) < max_peaks:
            i = int(np.flatnonzero(alive)[0])
            accepted.append(i)
            alive[i] = False
            q, h, s = (int(v) for v in qhs[i])
            width = _ridge_half_width(S[q], h, s, float(values[i]), cone_pad, footprint[1], N // 4)
            _, theta_edge = _adrt_to_hesse(q, h, s + width, N)
            sin_dw = abs(np.sin(float(theta_edge) - theta_all[i]))
            _, h_frame, s_frame = _hesse_to_adrt(rho_all, theta_all, N, quadrant=q)
            dh = np.abs(h_frame - h)
            ds = np.abs(s_frame - s)
            sin_d = np.abs(np.sin(theta_all - theta_all[i]))
            with np.errstate(divide="ignore"):
                envelope = sin_dw / sin_d * np.sqrt(length_all[i] / np.maximum(length_all, 1.0))
            envelope = np.minimum(1.0, wing_margin * envelope) * values[i] + wing_threshold
            in_wing = (dh <= ds + cone_pad) & (values <= envelope)
            alive &= ~in_wing

        # Stage 4: refinement, conversion, and seam de-duplication.
        lines: list[Line2D] = []
        for i in accepted:
            q, h, s = (int(v) for v in qhs[i])
            h_ref, s_ref = float(h), float(s)
            rho_theta: tuple[float, float] | None = None
            if rho_theta is None:
                h_ref = h + _parabolic_offset(S[q, :, s], h)
                s_ref = s + _parabolic_offset(S[q, h, :], s)
                rho_arr, theta_arr = _adrt_to_hesse(q, h_ref, s_ref, N)
                rho_theta = (float(rho_arr), float(theta_arr))
            rho, theta = rho_theta
            line = Line2D(rho, theta * geom.radians)
            if any(_same_line(line, l, dedup_rho, np.deg2rad(dedup_theta)) for l in lines):
                continue
            lines.append(line)

        self.log.debug(
            f"ADRT peaks: {n_local} local maxima, {n_contrast} after contrast test, "
            f"{len(accepted)} after suppression, {len(lines)} after de-duplication"
        )
        self.log.info(f"The Approximate Discrete Radon Transform detected {len(lines):d} line(s)")

        return [] if len(lines) == 0 else lines

    @timed("postprocess")
    def postprocess(
        self,
        streaks: afwTable.SourceTable,
        exposure: afwImage.ExposureF,
        lines: list[Line2D],
    ) -> NDArray[np.bool_]:
        """Perform postprocessing of detected linear features.

        Parameters
        ----------
        streaks : `lsst.afw.table.SourceTable`
            The output streak catalog.
        exposure : `lsst.afw.image.ExposureF`
            The exposure that was searched.
        lines : `list` [`Line2D`]
            The detected lines, in the PIXEL frame.

        Returns
        -------
        streak_mask : `numpy.ndarray`
            The streak mask plane array.
        """
        box = geom.Box2D(exposure.getBBox())
        wcs = exposure.getWcs()
        shape = exposure.image.array.shape
        line_masks = [np.zeros(shape, dtype=bool)]

        for line in lines:
            line_segment = line.clipped_to(box)
            if line_segment is None:
                continue

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(line_segment)

            center = line_segment.center
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()
            if wcs is not None:
                streak.setCoord(wcs.pixelToSky(center))

            # Individual streak mask
            line_mask = get_line_mask(line, shape, self.config.streak_width)
            line_masks.append(line_mask)
        self.log.info(f"Accepted {len(streaks):d} streak(s)")

        streak_mask = np.array(line_masks).any(axis=0)
        if self.config.only_mask_detected:
            streak_mask &= get_pixel_mask(exposure.mask, self.config.detected_mask_plane)

        return streak_mask


def _hesse_to_adrt(
    rho: NDArray[np.floating] | float,
    theta: NDArray[np.floating] | float,
    N: int,
    quadrant: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Hesse normal form parameters to ADRT coordinates.

    Analytic, vectorized map from Hesse normal ``(rho, theta)`` (in the ADRT's
    binned/padded pixel grid) to ADRT quadrant/height/slope indices
    ``(q, h, s)``. Returns floating-point indices so sub-pixel positions are
    preserved. This is the counterpart to the accumulator-side slope/intercept
    map in `_slope_intercept_map`; it places a known line's peak column, which
    the simulation-based validation tests use to seed `extract_segment_adrt`.

    ``theta`` is reduced modulo ``pi`` (Hesse lines are undirected). The
    round-trip is exact in the interior; it is degenerate only on the
    quadrant-boundary angles (0, 45, 90, 135 deg, i.e. ``s == 0`` and
    ``s == N - 1``), where adjacent ADRT quadrants share the same slope and the
    quadrant assignment is a convention choice.

    Parameters
    ----------
    rho, theta : `numpy.ndarray` or `float`
        The Hesse normal form rho (pixels) and theta (radians) parameters.
    N : `int`
        Size of the ADRT domain (must be a power of 2).
    quadrant : `int`, optional
        Force every line into the coordinate frame of this quadrant instead of
        the one its angle falls in. Lines outside the quadrant's 45-degree band
        then get extrapolated indices (``s`` outside ``[0, N-1]``), which is
        how a peak's neighbourhood is measured across a quadrant seam.

    Returns
    -------
    q, h, s : `numpy.ndarray`
        The ADRT quadrant, height, and slope indices, broadcast to the common
        shape of ``rho`` and ``theta``. ``q`` is integer-valued; ``h`` and
        ``s`` may be fractional.
    """
    rho = np.asarray(rho, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    c = (N - 1) / 2.0

    # Reduce to [0, pi) and pick the quadrant + base line angle. The four ADRT
    # quadrants tile [0, pi) in normal-vector angle as: [0, pi/4)->3,
    # [pi/4, pi/2)->2, [pi/2, 3pi/4)->1, [3pi/4, pi)->0.
    th = np.mod(theta, np.pi)
    if quadrant is None:
        conds = [th < np.pi / 4.0, th < np.pi / 2.0, th < 3.0 * np.pi / 4.0]
        q = np.select(conds, [3, 2, 1], default=0)
    else:
        # Bring theta to within half a turn of the quadrant's band center;
        # shifting by pi flips the sign of rho for the same line.
        center = (7 - 2 * quadrant) * np.pi / 8.0
        turns = np.round((center - th) / np.pi)
        th = th + turns * np.pi
        rho = rho * np.where(turns % 2 == 0, 1.0, -1.0)
        q = np.broadcast_to(np.asarray(quadrant), th.shape)
    ts = np.select(
        [q == 3, q == 2, q == 1],
        [th, np.pi / 2.0 - th, th - np.pi / 2.0],
        default=np.pi - th,
    )

    # Invert the slope geometry.
    ns = np.tan(ts)
    s = ns * (N - 1)
    cs = np.cos(ts) + np.sin(ts)

    # Invert the rho recenter/scale to recover the Radon offset, undo the
    # quadrant sign flip, then invert the height mapping. Trig uses `th` so it
    # is consistent with the quadrant reduction above.
    offset = (c * (np.cos(th) + np.sin(th)) - rho) / N
    h0 = np.where(q % 2 == 0, offset, -offset)
    hi = (h0 / cs + 0.5) * (1.0 + ns) - ((2.0 * N - 1.0) / (2.0 * N)) * ns
    h = N * (1.0 - hi) - 0.5

    return q.astype(np.float64), h, s


def _adrt_to_hesse(
    q: NDArray[np.integer] | int,
    h: NDArray[np.number] | float,
    s: NDArray[np.number] | float,
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ADRT coordinates to Hesse normal form parameters.

    Analytic, vectorized inverse of `_hesse_to_adrt`: maps ADRT
    quadrant/height/slope indices ``(q, h, s)`` to Hesse normal ``(rho, theta)``
    in the PIXEL frame of the grid the ADRT ran on (binned/padded). It
    reproduces `adrt.utils.coord_adrt` cell by cell without materializing the
    ``(4, 2N-1, N)`` coordinate arrays, and accepts fractional ``h`` and ``s``
    so sub-pixel peak positions convert directly.

    ``theta`` is returned in ``[0, pi)`` with ``rho`` sign-flipped accordingly,
    matching the `~astro_lfd.geom.Line2D` canonical form and the input
    convention of `_hesse_to_adrt`.

    Parameters
    ----------
    q : `numpy.ndarray` or `int`
        The ADRT quadrant index (0-3).
    h, s : `numpy.ndarray` or `float`
        The ADRT height and slope indices; may be fractional.
    N : `int`
        Size of the ADRT domain (must be a power of 2).

    Returns
    -------
    rho, theta : `numpy.ndarray`
        The Hesse normal form rho (pixels) and theta (radians, in ``[0, pi)``),
        broadcast to the common shape of the inputs.
    """
    q = np.asarray(q)
    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    c = (N - 1) / 2.0

    # Slope geometry: the digital line slope in the quadrant's local frame and
    # the ADRT height -> Radon offset map (the same algebra as in
    # `adrt.utils.coord_adrt`, here in closed form).
    ns = s / (N - 1)
    ts = np.arctan(ns)
    cs = np.cos(ts) + np.sin(ts)
    hi = 1.0 - (2.0 * h + 1.0) / (2.0 * N)
    h0 = ((hi + ((2.0 * N - 1.0) / (2.0 * N)) * ns) / (1.0 + ns) - 0.5) * cs
    offset = np.where(q % 2 == 0, h0, -h0)

    # Quadrant-dependent line angle, then to the normal angle and recenter the
    # offset from the image center to the PIXEL origin.
    angle = np.select(
        [q == 0, q == 1, q == 2, q == 3],
        [ts - np.pi / 2.0, -ts, ts, np.pi / 2.0 - ts],
    )
    theta = np.pi / 2.0 - angle
    rho = -offset * N + c * (np.cos(theta) + np.sin(theta))

    # Canonicalize to theta in [0, pi).
    theta = np.mod(theta, 2.0 * np.pi)
    flip = theta >= np.pi
    theta = np.where(flip, theta - np.pi, theta)
    rho = np.where(flip, -rho, rho)

    return rho, theta


def _ridge_half_width(
    S_q: NDArray[np.floating],
    h: int,
    s: int,
    value: float,
    pad: int,
    minimum: int,
    maximum: int,
) -> int:
    """Half-prominence half-width of a peak's ridge along the slope axis.

    Walks outwards from column ``s`` taking, at each slope distance ``d``, the
    maximum of the quadrant over the cone rows ``h +- (d + pad)``, and stops
    where that crest drops below half the peak value. The result is clipped
    to ``[minimum, maximum]``; the crest of a point-like source never drops,
    so ``maximum`` bounds the walk.
    """
    n_h, n_s = S_q.shape
    half = 0.5 * value
    for d in range(1, maximum):
        rows = slice(max(0, h - d - pad), min(n_h, h + d + pad + 1))
        crest = -np.inf
        for column in (s - d, s + d):
            if 0 <= column < n_s:
                crest = max(crest, float(S_q[rows, column].max()))
        if crest < half:
            return max(minimum, d)
    return maximum


def _parabolic_offset(profile: NDArray[np.floating], i: int) -> float:
    """Sub-cell offset of the maximum of ``profile`` near index ``i``.

    Fits a parabola through ``profile[i-1:i+2]``; returns 0 at the array
    edges or where the three points are not concave. Clipped to ``[-0.5, 0.5]``.
    """
    if i <= 0 or i >= len(profile) - 1:
        return 0.0
    y0, y1, y2 = (float(v) for v in profile[i - 1 : i + 2])
    denominator = y0 - 2.0 * y1 + y2
    if denominator >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))


def _same_line(a: Line2D, b: Line2D, rho_tol: float, theta_tol: float) -> bool:
    """Whether two canonical Hesse lines coincide within tolerance.

    Handles the wrap at ``theta = pi``, where ``(rho, theta)`` and
    ``(-rho, theta - pi)`` describe the same line.
    """
    dtheta = abs(a.theta.asRadians() - b.theta.asRadians())
    if dtheta > np.pi / 2.0:
        dtheta = np.pi - dtheta
        drho = abs(a.rho + b.rho)
    else:
        drho = abs(a.rho - b.rho)
    return dtheta < theta_tol and drho < rho_tol
