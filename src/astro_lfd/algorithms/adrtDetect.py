__all__ = ["ADRTDetectConfig", "ADRTDetectTask", "ADRTSegment", "extract_segment_adrt"]

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
    def detect(self, imarr: NDArray[np.float64]) -> list["ADRTSegment"]:
        """Perform linear feature detection on masked image array.

        Currently the peak detector is not fully implemented; it returns the
        global maximum for research and development. Each peak is passed to the
        closed-form butterfly analysis (`extract_segment_adrt`) to recover the
        line-segment moments (orientation, center, 2-D second moments) directly
        from the ADRT accumulator. Parameters are in the PIXEL frame.

        Parameters
        ----------
        imarr : `numpy.ndarray`
            The masked image array.

        Returns
        -------
        segments : `list` [`ADRTSegment`]
            The recovered segments, in the PIXEL frame.
        """
        adrt_result = adrt.adrt(imarr)
        N = adrt_result.shape[2]

        # Peak detector (to be developed fully). Returns integer accumulator
        # indices in the binned/padded grid the ADRT actually ran on.
        peaks = self._find_peaks(adrt_result)

        segments: list[ADRTSegment] = []
        for q, h, s in peaks:
            try:
                segment = extract_segment_adrt(adrt_result, int(q), float(h), int(s), N)
            except ValueError as error:
                self.log.warning("Skipping peak (q=%d, h=%d, s=%d): %s", q, h, s, error)
                continue
            segments.append(self._apply_bin_size(segment))

        return segments

    def _apply_bin_size(self, segment: "ADRTSegment") -> "ADRTSegment":
        """Rescale a binned-grid segment to full-resolution PIXEL coordinates.

        Binning is an isotropic rescale of the pixel grid: linear quantities
        (rho, center) scale by ``bin_size``, the 2-D second moments (which are in
        pixel^2) scale by ``bin_size**2``, and the angle ``theta`` is unchanged.
        This is done outside the ADRT coordinate transform, alongside the other
        array-frame -> PIXEL-frame corrections (e.g. a future ``XY0``).

        Parameters
        ----------
        segment : `ADRTSegment`
            The segment in the binned/padded grid.

        Returns
        -------
        scaled : `ADRTSegment`
            The segment rescaled to full-resolution pixels.
        """
        b = self.config.bin_size
        if b == 1:
            return segment

        return replace(
            segment,
            rho=segment.rho * b,
            center_x=segment.center_x * b,
            center_y=segment.center_y * b,
            mu20=segment.mu20 * b**2,
            mu11=segment.mu11 * b**2,
            mu02=segment.mu02 * b**2,
        )

    @timed("postprocess")
    def postprocess(
        self,
        streaks: afwTable.SourceTable,
        exposure: afwImage.ExposureF,
        *,
        segments: list["ADRTSegment"],
    ) -> None:
        """Perform postprocessing of detected linear features.

        Builds the finite line-segment representation of each detected streak
        from the extracted moments (center, orientation, and the top-hat
        length/width from `ADRTSegment.segment_dimensions`) and stores it in the
        streak catalog, together with the recovered width. The segment is clipped
        to the exposure bounding box. For ADRT the Hesse normal origin is the
        PIXEL origin, so no translation is needed.

        Parameters
        ----------
        streaks : `lsst.afw.table.SourceTable`
            The output streak catalog.
        exposure : `lsst.afw.image.ExposureF`
            The exposure that was searched.
        segments : `list` [`ADRTSegment`]
            The recovered segments, in the PIXEL frame.
        """
        if not segments:
            return

        box = geom.Box2D(exposure.getBBox())
        wcs = exposure.getWcs()
        for segment in segments:
            line = Line2D(segment.rho, segment.theta * geom.radians)

            # Derive the top-hat length/width from the moment tensor for the
            # finite-segment representation, then build the segment from the
            # recovered center and length and clip it to the frame. The
            # along-line center coordinate is the center point projected onto the
            # line direction.
            length, width, _ = segment.segment_dimensions()
            center = geom.Point2D(segment.center_x, segment.center_y)
            s_center = line.along_coordinate(center)
            line_segment = LineSegment2D.from_center_length(line, s_center, length)

            clipped = line_segment.clipped_to(box)
            if clipped is None:
                continue

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(clipped)
            streak["line_width"] = width

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
        analysis (`extract_segment_adrt`) consumes directly. Focus on
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


def _hesse_to_adrt(
    rho: NDArray[np.floating] | float,
    theta: NDArray[np.floating] | float,
    N: int,
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
class ADRTSegment:
    """Closed-form line-segment moments from the ADRT accumulator.

    Result of `extract_segment_adrt`. The primary data products are the image
    moments: the center (first moments) and the 2-D central second moments
    ``(mu20, mu11, mu02)``, which are identical to the ``(XX, XY, YY)`` moments
    used to characterize sources in LSST catalogs. All quantities are in the
    pixel grid the ADRT ran on (binned/padded); the caller undoes binning (see
    `ADRTDetectTask.detect`). See ``docs/detectors/adrt/butterfly.md`` for the
    derivation.

    For a top-hat (uniform rectangle) model — useful for simulations and
    validation — call `segment_dimensions` to invert the moment tensor to
    ``(length, width, phi0)``.

    Attributes
    ----------
    rho, theta : `float`
        Refined Hesse normal form parameters (pixels, radians). ``theta`` is the
        line-normal angle; the orientation follows from the fitted moment tensor
        rather than the integer peak column.
    center_x, center_y : `float`
        Segment center in the ADRT pixel grid (pixels), from the centroid fit.
    mu20, mu11, mu02 : `float`
        The 2-D central second moments of the feature (pixels^2), equal to the
        catalog ``(XX, XY, YY)``. Related to the fitted variance quadratic
        ``V(s) = A s^2 + B s + C`` by ``(A, B, C) = (mu20, -2 mu11, mu02)``.
    var_residual : `float`
        RMS residual of the variance quadratic fit (pixels^2), a fit-quality
        diagnostic.
    n_columns : `int`
        Number of slope columns used in the fit.
    """

    rho: float
    theta: float
    center_x: float
    center_y: float
    mu20: float
    mu11: float
    mu02: float
    var_residual: float
    n_columns: int

    def segment_dimensions(self, width_bias: float = 0.0) -> tuple[float, float, float]:
        """Invert the moment tensor to top-hat ``(length, width, phi0)``.

        Treats the feature as a uniform rectangle and inverts its 2-D inertia
        tensor ``(mu20, mu11, mu02)`` to the longitudinal length, transverse
        width, and line angle. This is the top-hat-specific interpretation of the
        moments (see `_invert_inertia`); for arbitrary streak morphologies the
        moments themselves are the primary description.

        Parameters
        ----------
        width_bias : `float`, optional
            Additive discretization bias in ``w^2`` (pixels^2) to subtract before
            taking the root (0.0, by default). See the width-bias discussion in
            ``docs/detectors/adrt/butterfly.md``.

        Returns
        -------
        length, width : `float`
            The recovered top-hat length and width (pixels).
        phi0 : `float`
            The line angle ``phi0`` (radians), ``dy/dx = tan(phi0)``.
        """
        return _invert_inertia(self.mu20, -2.0 * self.mu11, self.mu02, width_bias=width_bias)


def _slope_intercept_map(
    q: int,
    s_cols: NDArray[np.integer],
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Closed-form slope value and affine height->intercept map per column.

    For a fixed ADRT quadrant/slope column the continuous line slope
    ``s = dy/dx`` is constant, and the intercept ``b = y - s x`` is an *exact
    affine* function of the integer height index ``h``: ``b = alpha * h + beta``.
    Both follow in closed form from the ADRT digital-line geometry (verified to
    floating-point precision against the former per-cell coordinate transform),
    so the whole slope band is handled with three short vectorized arrays and no
    per-row transform or per-column Python loop.

    - The slope is the quadrant-selected map of the slope index (the four
      combinations of ``+/- s/(N-1)`` and ``+/- (N-1)/s``).
    - ``(alpha, beta)`` come from the same geometry; only the intercept axis is
      rescaled per column, which does not change that the centroid is linear and
      the variance quadratic in the slope (see
      ``docs/detectors/adrt/butterfly.md``).

    Parameters
    ----------
    q : `int`
        The quadrant index of the columns (0-3).
    s_cols : `numpy.ndarray` of `int`
        The integer slope (column) indices, strictly interior to ``[1, N-1)``.
    N : `int`
        The ADRT domain size (power of two).

    Returns
    -------
    slopes : `numpy.ndarray`
        The continuous line slope ``dy/dx`` for each column.
    alpha, beta : `numpy.ndarray`
        The affine intercept-map coefficients, ``b = alpha * h + beta``.
    """
    s = s_cols.astype(np.float64)
    c = (N - 1) / 2.0
    k = (2.0 * N - 1.0) / (2.0 * N)

    ns = s / (N - 1)
    ts = np.arctan(ns)
    cs = np.cos(ts) + np.sin(ts)

    # Normal-vector angle theta = pi/2 - line-angle; the quadrant sign s_q flips
    # the offset for the odd quadrants. See _hesse_to_adrt for the inverse map.
    angle = {
        0: ts - np.pi / 2.0,
        1: -ts,
        2: ts,
        3: np.pi / 2.0 - ts,
    }[q]
    theta = np.pi / 2.0 - angle
    sin_t = np.sin(theta)
    s_q = 1.0 if q % 2 == 0 else -1.0

    slopes = -np.cos(theta) / sin_t

    # Affine intercept map: b = (P * h + Q) / sin(theta), with P, Q from the
    # ADRT height->offset->rho chain (constant across rows of a column).
    P = s_q * cs / (1.0 + ns)
    Q = c * (np.cos(theta) + np.sin(theta)) - s_q * N * cs * (k - 0.5)
    alpha = P / sin_t
    beta = Q / sin_t

    return slopes, alpha, beta


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


def extract_segment_adrt(
    adrt_result: NDArray[np.float64],
    q: int,
    h: float,
    s_idx: int,
    N: int,
    *,
    half_band: int = 90,
    background: str = "median",
) -> ADRTSegment:
    """Extract line-segment moments directly from the ADRT accumulator.

    The ADRT "butterfly" analysis. For a fixed slope column the accumulator is
    the image projected onto the intercept axis ``b = y - s x``, so the
    accumulator-weighted column centroid and variance are exactly the image
    moments: ``mu(s) = y_c - s x_c`` (linear) and
    ``V(s) = mu20 s^2 - 2 mu11 s + mu02`` (quadratic in the continuous slope).
    Fitting both recovers the center and the 2-D central second moments
    ``(mu20, mu11, mu02)`` (catalog ``XX, XY, YY``) in closed form. Derivation
    and validation: ``docs/detectors/adrt/butterfly.md``.

    The slope value and the height->intercept map are both closed-form functions
    of the quadrant and slope index (`_slope_intercept_map`), so the whole slope
    band is processed with vectorized array operations — no per-cell coordinate
    transform and no Python loop over columns.

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
        default). The variance law is globally quadratic, so the result is
        insensitive to this within one quadrant.
    background : `str`, optional
        Per-column background model subtracted before the moment sums.
        ``"median"`` (default) subtracts the per-column median (clipped at zero);
        ``"none"`` subtracts nothing.

    Returns
    -------
    segment : `ADRTSegment`
        The recovered moments and fit diagnostics, in the ADRT pixel grid
        (binning undone by the caller).

    Raises
    ------
    ValueError
        Raised if the background model is unknown, or if fewer than three usable
        slope columns fall within the band and the peak's quadrant, so the
        quadratic fit is under-determined.
    """
    if background not in ("median", "none"):
        raise ValueError(f"unknown background model: {background!r}")

    # Slope band clipped to the peak's quadrant (columns 0 and N-1 are the
    # quadrant-boundary slopes; stay strictly interior to avoid the seam).
    lo = max(1, s_idx - half_band)
    hi = min(N - 1, s_idx + half_band)
    s_cols = np.arange(lo, hi)

    slopes, alpha, beta = _slope_intercept_map(q, s_cols, N)

    # Per-column accumulator weights over the full height axis, background
    # subtracted per column. Columns are the trailing axis so moment sums reduce
    # over axis 0 (the height/intercept axis).
    weights = adrt_result[q, :, lo:hi].astype(np.float64)  # (2N-1, n_cols)
    if background == "median":
        weights = weights - np.median(weights, axis=0, keepdims=True)
        np.clip(weights, 0.0, None, out=weights)

    total = weights.sum(axis=0)
    good = total > 0
    if int(good.sum()) < 3:
        raise ValueError(
            f"only {int(good.sum())} usable slope column(s) in band; need >= 3 for the quadratic fit"
        )

    slopes, alpha, beta = slopes[good], alpha[good], beta[good]
    weights, total = weights[:, good], total[good]

    # Weighted moments of the integer height index per column, then mapped to the
    # continuous intercept b = alpha h + beta: centroid(b) = alpha <h> + beta and
    # Var(b) = alpha^2 Var(h). The affine map is exact, so these are the physical
    # image moments with no approximation.
    h_idx = np.arange(weights.shape[0], dtype=np.float64)[:, None]
    mean_h = (weights * h_idx).sum(axis=0) / total
    var_h = (weights * (h_idx - mean_h) ** 2).sum(axis=0) / total
    centroid = alpha * mean_h + beta
    variance = alpha**2 * var_h

    # Weighted fits: variance quadratic V(s) = A s^2 + B s + C and centroid line
    # mu(s) = beta0 + beta1 s. Weight by column flux so the peak dominates and
    # far, contaminated columns matter less.
    sqrt_w = np.sqrt(total)
    C_coef, B_coef, A_coef = (float(c) for c in Polynomial.fit(slopes, variance, deg=2, w=sqrt_w).convert().coef)
    beta0, beta1 = (float(c) for c in Polynomial.fit(slopes, centroid, deg=1, w=sqrt_w).convert().coef)

    # The quadratic coefficients are the central second-moment (inertia) tensor:
    # (A, B, C) = (mu20, -2 mu11, mu02).
    mu20 = A_coef
    mu11 = -0.5 * B_coef
    mu02 = C_coef

    # Position: mu(s) = y_c - s x_c, so beta1 = -x_c and beta0 = y_c.
    center_x = -beta1
    center_y = beta0

    # Refine (rho, theta) from the fitted moment tensor rather than the integer
    # peak. The major-axis (line) angle is phi0 = 1/2 atan2(2 mu11, mu20 - mu02);
    # the normal angle is phi0 + pi/2 and rho is the center projected on it.
    phi0 = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    theta = phi0 + np.pi / 2.0
    rho = center_x * np.cos(theta) + center_y * np.sin(theta)

    # Fit-quality diagnostic: RMS residual of the variance quadratic.
    model = A_coef * slopes**2 + B_coef * slopes + C_coef
    var_residual = float(np.sqrt(np.mean((variance - model) ** 2)))

    return ADRTSegment(
        rho=float(rho),
        theta=float(theta),
        center_x=float(center_x),
        center_y=float(center_y),
        mu20=float(mu20),
        mu11=float(mu11),
        mu02=float(mu02),
        var_residual=var_residual,
        n_columns=int(slopes.size),
    )
