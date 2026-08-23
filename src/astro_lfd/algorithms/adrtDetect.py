__all__ = ["ADRTDetectConfig", "ADRTDetectTask", "ADRTSegmentEstimate", "estimate_segment_adrt"]

from dataclasses import dataclass, replace

import adrt
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from .base import binary_dilation, get_pixel_mask, timed
from ..geom.line import Line2D, LineSegment2D
from ..table.streakAdapter import StreakAdapter


class ADRTDetectConfig(pexConfig.Config):
    """Configurable parameters for `ADRTDetectTask`."""

    # Configuration for preprocessing
    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    bin_size = pexConfig.Field(
        doc="Pixel bin size for input image.",
        dtype=int,
        default=1,
    )


class ADRTDetectTask(pipeBase.Task):
    """Detect linear features with the Approximate Discrete Radon Transform."""

    ConfigClass = ADRTDetectConfig
    _DefaultName = "adrtDetect"

    timings: dict[str, float]

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
            ``timings``
                Computing times for each processing step (`dict`)
        """
        streaks = afwTable.SourceCatalog(table)
        self.timings = {}
        imarr = self.preprocess(exposure)
        segments = self.detect(imarr)
        self.postprocess(streaks, exposure, segments=segments)

        return pipeBase.Struct(streaks=streaks, imarr=imarr, timings=self.timings)

    @timed("preprocess")
    def preprocess(self, exposure: afwImage.ExposureF) -> NDArray[np.float64]:
        """Perform preprocessing on input exposure.

        Parameters
        ----------
        exposure: `lsst.afw.image.ExposureF`
            The exposure to search.

        Returns
        -------
        imarr : `numpy.ndarray`
            The image array with invalid regions masked.
        """
        mi = afwMath.binImage(exposure.maskedImage, self.config.bin_size)
        imarr = mi.image.array

        bad_mask = get_pixel_mask(mi.mask, self.config.bad_mask_planes)
        if self.config.bin_size == 1:
            bad_mask = binary_dilation(bad_mask, 1)
        imarr[bad_mask] = 0.0

        padded_imarr = np.pad(
            imarr,
            ((0, 4096 // self.config.bin_size - imarr.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        return padded_imarr

    @timed("detect")
    def detect(self, imarr: NDArray[np.float64]) -> list["ADRTSegmentEstimate"]:
        """Perform linear feature detection on masked image array.

        Currently the peak detector is not fully implemented; it returns the
        global maximum for research and development. Each peak is passed to the
        closed-form butterfly analysis (`estimate_segment_adrt`) to recover the
        full segment geometry (orientation, length, width, center) directly from
        the ADRT accumulator. Parameters are in the PIXEL frame.

        Parameters
        ----------
        imarr : `numpy.ndarray`
            The masked image array.

        Returns
        -------
        segments : `list` [`ADRTSegmentEstimate`]
            The recovered segment estimates, in the PIXEL frame.
        """
        adrt_result = adrt.adrt(imarr)
        N = adrt_result.shape[2]

        # Peak detector (to be developed fully). Returns integer accumulator
        # indices in the binned/padded grid the ADRT actually ran on.
        peaks = self._find_peaks(adrt_result)

        segments: list[ADRTSegmentEstimate] = []
        for q, h, s in peaks:
            try:
                estimate = estimate_segment_adrt(adrt_result, int(q), float(h), int(s), N)
            except ValueError as error:
                self.log.warning("Skipping peak (q=%d, h=%d, s=%d): %s", q, h, s, error)
                continue
            segments.append(self._apply_bin_size(estimate))

        return segments

    def _apply_bin_size(self, estimate: "ADRTSegmentEstimate") -> "ADRTSegmentEstimate":
        """Rescale a binned-grid estimate to full-resolution PIXEL coordinates.

        Binning is an isotropic rescale of every length (rho, length, width,
        center) by ``bin_size`` and leaves angle/slope unchanged. This is done
        outside the ADRT coordinate transform, alongside the other array-frame
        -> PIXEL-frame corrections (e.g. a future ``XY0``).

        Parameters
        ----------
        estimate : `ADRTSegmentEstimate`
            The estimate in the binned/padded grid.

        Returns
        -------
        scaled : `ADRTSegmentEstimate`
            The estimate rescaled to full-resolution pixels.
        """
        b = self.config.bin_size
        if b == 1:
            return estimate

        return replace(
            estimate,
            rho=estimate.rho * b,
            length=estimate.length * b,
            width=estimate.width * b,
            center_x=estimate.center_x * b,
            center_y=estimate.center_y * b,
        )

    @timed("postprocess")
    def postprocess(
        self,
        streaks: afwTable.SourceTable,
        exposure: afwImage.ExposureF,
        *,
        segments: list["ADRTSegmentEstimate"],
    ) -> None:
        """Perform postprocessing of detected linear features.

        Builds the finite line-segment representation of each detected streak
        from the butterfly estimate (center, orientation, length) and stores it
        in the streak catalog, together with the recovered width. The segment is
        clipped to the exposure bounding box. For ADRT the Hesse normal origin
        is the PIXEL origin, so no translation is needed.

        Parameters
        ----------
        streaks : `lsst.afw.table.SourceTable`
            The output streak catalog.
        exposure : `lsst.afw.image.ExposureF`
            The exposure that was searched.
        segments : `list` [`ADRTSegmentEstimate`]
            The recovered segment estimates, in the PIXEL frame.
        """
        if not segments:
            return

        box = geom.Box2D(exposure.getBBox())
        wcs = exposure.getWcs()
        for estimate in segments:
            line = Line2D(estimate.rho, estimate.theta * geom.radians)

            # Build the finite segment from the recovered center and length,
            # then clip it to the frame. The along-line center coordinate is the
            # center point projected onto the line direction.
            center = geom.Point2D(estimate.center_x, estimate.center_y)
            s_center = line.along_coordinate(center)
            segment = LineSegment2D.from_center_length(line, s_center, estimate.length)

            clipped = segment.clipped_to(box)
            if clipped is None:
                continue

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(clipped)
            streak["line_width"] = estimate.width

            streak_center = clipped.center
            streak["line_center_x"] = streak_center.getX()
            streak["line_center_y"] = streak_center.getY()
            if wcs is not None:
                streak.setCoord(wcs.pixelToSky(streak_center))

        self.log.info("Accepted %d streak(s) from ADRT butterfly analysis", len(streaks))

    def _find_peaks(
        self,
        adrt_result: NDArray[np.float64],
    ) -> list[tuple[int, int, int]]:
        """Placeholder for peak finding in the ADRT transform space.

        Will eventually detect multiple peaks. For now it returns the single
        global maximum as integer accumulator indices, which the butterfly
        analysis (`estimate_segment_adrt`) consumes directly. Focus on
        implementation first, then decide optimizations (within Python or as
        an extension to a branch of `adrt` if C++ implementation needed).

        Parameters
        ----------
        adrt_result : `numpy.ndarray`
            The ADRT result.

        Returns
        -------
        peaks : `list` [`tuple` [`int`, `int`, `int`]]
            The detected peaks as ``(q, h, s)`` accumulator indices (quadrant,
            height, slope).
        """
        # Get global maximum indices (placeholder for multipeak finding).
        q, h, s = np.unravel_index(np.argmax(adrt_result), adrt_result.shape)

        return [(int(q), int(h), int(s))]


def _adrt_to_hesse(
    q: NDArray[np.integer] | int,
    h: NDArray[np.floating] | float,
    s: NDArray[np.floating] | float,
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ADRT coordinates to Hesse normal form parameters.

    This is the analytic, vectorized inverse of `_hesse_to_adrt`. It maps the
    ADRT quadrant/height/slope indices ``(q, h, s)`` returned by a peak finder
    directly to Hesse normal ``(rho, theta)`` parameters, reproducing
    `adrt.utils.coord_adrt` in closed form (to floating-point precision) rather
    than building and indexing its full coordinate tables. Being analytic, it
    also accepts *fractional* ``h`` and ``s`` from sub-pixel peak refinement,
    which the integer-indexed table lookup cannot.

    The returned parameters are in the pixel grid the ADRT ran on (the binned,
    zero-padded array). This utility is intentionally agnostic to preprocessing:
    undoing binning and applying any ``XY0`` origin offset to reach the true
    LSST PIXEL frame is the caller's responsibility (see `detect`). Padding is
    already accounted for via ``N`` and only enlarges the domain along the
    bottom rows, so it does not move the origin.

    Parameters
    ----------
    q : `numpy.ndarray` or `int`
        The ADRT result quadrant index/indices (0-3).
    h, s : `numpy.ndarray` or `float`
        The ADRT result height and slope index/indices. May be fractional
        (sub-pixel).
    N : `int`
        Size of the ADRT domain (must be a power of 2).

    Returns
    -------
    rho, theta : `numpy.ndarray`
        The Hesse normal form rho (pixels) and theta (radians) parameters,
        broadcast to the common shape of ``q``, ``h``, ``s``.
    """
    q = np.asarray(q)
    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    c = (N - 1) / 2.0

    # Base digital-line geometry from the slope index (coord_adrt internals).
    ns = s / (N - 1)
    ts = np.arctan(ns)
    cs = np.cos(ts) + np.sin(ts)

    # Fractional Radon-domain offset from the height index.
    hi = 1.0 - (2.0 * h + 1.0) / (2.0 * N)
    h0 = ((hi + ((2.0 * N - 1.0) / (2.0 * N)) * ns) / (1.0 + ns) - 0.5) * cs

    # Quadrant selects the sign of the offset and the line angle. Even
    # quadrants (0, 2) keep the offset sign; odd quadrants (1, 3) flip it.
    offset = np.where(q % 2 == 0, h0, -h0)
    angle = np.select(
        [q == 0, q == 1, q == 2, q == 3],
        [ts - np.pi / 2.0, -ts, ts, np.pi / 2.0 - ts],
    )

    # Line angle -> normal-vector angle, then Radon offset -> PIXEL rho with the
    # image-center -> corner-origin recenter (see devel/adrt_coordinate_transform.md).
    theta = np.pi / 2.0 - angle
    rho = -offset * N + c * (np.cos(theta) + np.sin(theta))

    return rho, theta


def _hesse_to_adrt(
    rho: NDArray[np.floating] | float,
    theta: NDArray[np.floating] | float,
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Hesse normal form parameters to ADRT coordinates.

    Analytic, vectorized inverse of `_adrt_to_hesse`: maps Hesse normal
    ``(rho, theta)`` (in the ADRT's binned/padded pixel grid) back to ADRT
    quadrant/height/slope indices ``(q, h, s)``. Returns floating-point indices
    so sub-pixel positions are preserved.

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
    conds = [th < np.pi / 4.0, th < np.pi / 2.0, th < 3.0 * np.pi / 4.0]
    q = np.select(conds, [3, 2, 1], default=0)
    ts = np.select(
        conds,
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


@dataclass
class ADRTSegmentEstimate:
    """Closed-form line-segment geometry from the ADRT accumulator.

    Result of `estimate_segment_adrt`. All lengths are in the pixel grid the
    ADRT ran on (binned/padded); the caller undoes binning (see
    `ADRTDetectTask.detect`). See ``docs/detectors/adrt/butterfly.md`` for the
    derivation.

    Attributes
    ----------
    rho, theta : `float`
        Refined Hesse normal form parameters (pixels, radians). ``theta`` is the
        line-normal angle; the segment orientation follows from the fitted
        moment tensor rather than the integer peak column.
    length, width : `float`
        Top-hat longitudinal length and transverse width (pixels), from the
        principal moments of the fitted variance quadratic.
    center_x, center_y : `float`
        Segment center in the ADRT pixel grid (pixels).
    slope : `float`
        Line slope ``s0 = dy/dx = tan(phi0)``.
    A, B, C : `float`
        Fitted variance-quadratic coefficients ``V(s) = A s^2 + B s + C``, equal
        to the image central second moments ``(mu20, -2 mu11, mu02)``.
    var_residual : `float`
        RMS residual of the variance quadratic fit (pixels^2), a fit-quality
        diagnostic.
    n_columns : `int`
        Number of slope columns used in the fit.
    """

    rho: float
    theta: float
    length: float
    width: float
    center_x: float
    center_y: float
    slope: float
    A: float
    B: float
    C: float
    var_residual: float
    n_columns: int


def _column_moments_continuous(
    adrt_result: NDArray[np.float64],
    q: int,
    s_idx: int,
    N: int,
    background: str = "median",
) -> tuple[float, float, float, float]:
    """Return continuous-coordinate moments of one ADRT slope column.

    Maps every row of the column through `_adrt_to_hesse` to the continuous
    line slope ``s = dy/dx`` and intercept ``b = y - s x``, subtracts a
    per-column background, and forms the accumulator-weighted centroid and
    variance of ``b``. This is the exact continuous parameterization the
    variance law is derived in (see ``docs/detectors/adrt/butterfly.md``);
    fitting against raw integer indices is only locally quadratic.

    Parameters
    ----------
    adrt_result : `numpy.ndarray`, (4, 2N-1, N)
        The ADRT result.
    q : `int`
        The quadrant index of the column.
    s_idx : `int`
        The integer slope (column) index.
    N : `int`
        The ADRT domain size.
    background : `str`, optional
        Per-column background model subtracted before the moment sums.
        ``"median"`` (default) subtracts the column median (clipped at zero);
        ``"none"`` subtracts nothing.

    Returns
    -------
    s_cont : `float`
        The continuous line slope for this column.
    centroid, variance : `float`
        The weighted centroid and variance of the intercept ``b``. Both are
        ``nan`` if the column has no positive weight.
    total : `float`
        Total (background-subtracted) weight in the column, usable as a fit
        weight.
    """
    n_rows = adrt_result.shape[1]  # 2N - 1
    h_idx = np.arange(n_rows, dtype=np.float64)

    rho, theta = _adrt_to_hesse(q, h_idx, float(s_idx), N)
    theta_col = float(theta.reshape(-1)[0])
    sin_t = np.sin(theta_col)

    # Line slope dy/dx and intercept b = y - s x from Hesse normal form. Rows of
    # a fixed column share a single slope; only the intercept varies.
    s_cont = -np.cos(theta_col) / sin_t
    b = rho / sin_t

    weights = adrt_result[q, :, s_idx].astype(np.float64)
    if background == "median":
        weights = weights - np.median(weights)
        np.clip(weights, 0.0, None, out=weights)
    elif background != "none":
        raise ValueError(f"unknown background model: {background!r}")

    total = weights.sum()
    if total <= 0:
        return s_cont, np.nan, np.nan, 0.0

    centroid = float(np.sum(weights * b) / total)
    variance = float(np.sum(weights * (b - centroid) ** 2) / total)
    return s_cont, centroid, variance, float(total)


def _invert_inertia(A: float, B: float, C: float, width_bias: float = 0.0) -> tuple[float, float, float]:
    """Invert variance-quadratic coefficients to ``(length, width, phi0)``.

    The coefficients ``V(s) = A s^2 + B s + C`` are the image central second
    moments ``(mu20, -2 mu11, mu02)`` (the 2-D inertia tensor). Its eigenvalues
    are the principal moments ``L^2/12`` (major) and ``w^2/12`` (minor) of a
    uniform rectangle, and its major-axis orientation is the line angle. See
    ``docs/detectors/adrt/butterfly.md`` sec. 3.

    Parameters
    ----------
    A, B, C : `float`
        The variance-quadratic coefficients.
    width_bias : `float`, optional
        Additive discretization bias in ``w^2`` (pixels^2) to subtract before
        taking the root (0.0, by default). See the width-bias discussion in the
        derivation doc.

    Returns
    -------
    length, width : `float`
        The recovered top-hat length and width (pixels). Non-negative;
        clamped at zero if the (bias-corrected) principal moment is negative.
    phi0 : `float`
        The line angle ``phi0`` (radians), ``dy/dx = tan(phi0)``.
    """
    half_sum = A + C
    root = float(np.hypot(A - C, B))  # sqrt((A - C)^2 + B^2)
    length_sq = 6.0 * (half_sum + root)
    width_sq = 6.0 * (half_sum - root) - width_bias
    phi0 = 0.5 * float(np.arctan2(-B, A - C))
    return np.sqrt(max(length_sq, 0.0)), np.sqrt(max(width_sq, 0.0)), phi0


def estimate_segment_adrt(
    adrt_result: NDArray[np.float64],
    q: int,
    h: float,
    s_idx: int,
    N: int,
    *,
    half_band: int = 90,
    background: str = "median",
    width_bias: float = 0.0,
) -> ADRTSegmentEstimate:
    """Estimate line-segment geometry directly from the ADRT accumulator.

    The ADRT "butterfly" analysis. For a fixed slope column the accumulator is
    the image projected onto the intercept axis ``b = y - s x``, so the
    accumulator-weighted column centroid and variance are exactly the image
    moments: ``mu(s) = y_c - s x_c`` (linear) and
    ``V(s) = mu20 s^2 - 2 mu11 s + mu02`` (quadratic in the continuous slope).
    Fitting both and inverting the variance quadratic recovers the full segment
    geometry in closed form. Derivation and validation:
    ``docs/detectors/adrt/butterfly.md``.

    Parameters
    ----------
    adrt_result : `numpy.ndarray`, (4, 2N-1, N)
        The ADRT result.
    q : `int`
        The quadrant index of the detected peak.
    h : `float`
        The height index of the detected peak. Accepted so the natural peak
        descriptor ``(q, h, s_idx)`` can be passed straight through; the current
        moment analysis spans the full height axis of each column and does not
        use it. Reserved for future height-windowing around the ridge.
    s_idx : `int`
        The integer slope (column) index of the detected peak.
    N : `int`
        The ADRT domain size (power of two).
    half_band : `int`, optional
        Number of slope columns to include on each side of the peak (90, by
        default). The variance law is globally quadratic, so the estimate is
        insensitive to this within one quadrant.
    background : `str`, optional
        Per-column background model (``"median"`` or ``"none"``); see
        `_column_moments_continuous`.
    width_bias : `float`, optional
        Additive discretization bias in ``w^2`` (pixels^2) to subtract; see
        `_invert_inertia` (0.0, by default).

    Returns
    -------
    estimate : `ADRTSegmentEstimate`
        The recovered segment geometry and fit diagnostics, in the ADRT pixel
        grid (binning undone by the caller).

    Raises
    ------
    ValueError
        Raised if fewer than three usable slope columns fall within the band and
        the peak's quadrant, so the quadratic fit is under-determined.
    """
    # Slope band clipped to the peak's quadrant (columns 0 and N-1 are the
    # quadrant-boundary slopes; stay strictly interior to avoid the seam).
    lo = max(1, s_idx - half_band)
    hi = min(N - 1, s_idx + half_band)
    s_cols = np.arange(lo, hi)

    s_cont = np.empty(s_cols.size)
    centroid = np.empty(s_cols.size)
    variance = np.empty(s_cols.size)
    weight = np.empty(s_cols.size)
    for i, col in enumerate(s_cols):
        s_cont[i], centroid[i], variance[i], weight[i] = _column_moments_continuous(
            adrt_result, q, int(col), N, background=background
        )

    good = np.isfinite(variance) & np.isfinite(centroid) & (weight > 0)
    if good.sum() < 3:
        raise ValueError(
            f"only {int(good.sum())} usable slope column(s) in band; need >= 3 for the quadratic fit"
        )
    s_cont, centroid, variance, weight = s_cont[good], centroid[good], variance[good], weight[good]

    # Weighted fits: variance quadratic V(s) = A s^2 + B s + C and centroid line
    # mu(s) = beta0 + beta1 s. Weight by column flux so the peak dominates and
    # far, contaminated columns matter less.
    sqrt_w = np.sqrt(weight)
    C_coef, B_coef, A_coef = (float(c) for c in Polynomial.fit(s_cont, variance, deg=2, w=sqrt_w).convert().coef)
    beta0, beta1 = (float(c) for c in Polynomial.fit(s_cont, centroid, deg=1, w=sqrt_w).convert().coef)

    length, width, phi0 = _invert_inertia(A_coef, B_coef, C_coef, width_bias=width_bias)

    # Position: mu(s) = y_c - s x_c, so beta1 = -x_c and beta0 = y_c.
    center_x = -beta1
    center_y = beta0

    # Refine (rho, theta) from the fitted geometry rather than the integer peak.
    # The normal angle is phi0 + pi/2; rho is the center projected on the normal.
    theta = phi0 + np.pi / 2.0
    rho = center_x * np.cos(theta) + center_y * np.sin(theta)

    # Fit-quality diagnostic: RMS residual of the variance quadratic.
    model = A_coef * s_cont**2 + B_coef * s_cont + C_coef
    var_residual = float(np.sqrt(np.mean((variance - model) ** 2)))

    return ADRTSegmentEstimate(
        rho=float(rho),
        theta=float(theta),
        length=float(length),
        width=float(width),
        center_x=float(center_x),
        center_y=float(center_y),
        slope=float(np.tan(phi0)),
        A=float(A_coef),
        B=float(B_coef),
        C=float(C_coef),
        var_residual=var_residual,
        n_columns=int(s_cont.size),
    )
