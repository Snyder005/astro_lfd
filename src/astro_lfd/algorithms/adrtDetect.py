__all__ = ["ADRTDetectConfig", "ADRTDetectTask", "AdrtPeak"]

from dataclasses import dataclass

import adrt
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .adrtButterfly import _adrt_to_hesse, _hesse_to_adrt, extract_segment_adrt
from .base import binary_dilation, get_line_mask, get_pixel_mask, timed
from ..geom.line import Line2D
from ..table.streakAdapter import StreakAdapter


@dataclass(frozen=True)
class AdrtPeak:
    """A detected peak in the ADRT accumulator.

    Attributes
    ----------
    q, h, s : `int`
        Accumulator indices (quadrant, height, slope) of the local maximum.
    h_refined, s_refined : `float`
        Sub-cell height and slope after refinement (equal to ``h``, ``s`` when
        no refinement was requested).
    value : `float`
        Significance of the peak: the accumulator value divided by the square
        root of the digital line length, in units of the noise standard
        deviation when the input is unit-variance white noise.
    line : `~astro_lfd.geom.Line2D`
        The line in the PIXEL frame of the grid the ADRT ran on.
    """

    q: int
    h: int
    s: int
    h_refined: float
    s_refined: float
    value: float
    line: Line2D


class ADRTDetectConfig(pexConfig.Config):
    """Configurable parameters for `ADRTDetectTask`."""

    # Configuration for preprocess
    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    mask_edge_pixels = pexConfig.Field(
        doc="Number of pixels to mask around image edges.",
        dtype=int,
        default=15,
    )

    # Configuration for postprocess


class ADRTDetectTask(pipeBase.Task):
    """Detect linear features with the Approximate Discrete Radon Transform."""

    ConfigClass = ADRTDetectConfig
    _DefaultName = "adrtDetect"

    timings: dict[str, float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = {}
        # Digital line length per accumulator cell, keyed by the ADRT domain
        # size. One extra forward transform per N, reused across exposures.
        self._line_length_cache: dict[int, NDArray[np.float32]] = {}

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
        imarr = self.preprocess(exposure)
        lines = self.detect(imarr)
        streak_mask = self.postprocess(streaks, exposure, lines)

        return pipeBase.Struct(
            streaks=streaks,
            imarr=imarr,
            streak_mask=streak_mask,
            timings=self.timings,
        )

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
        imarr = exposure.image.array.copy()

        bad_mask = get_pixel_mask(mi.mask, self.config.bad_mask_planes)
        if edge := self.config.mask_edge_pixels:
            bad_mask[:edge, :] = True
            bad_mask[-edge:, :] = True
            bad_mask[:, :edge] = True
            bad_mask[:, -edge:] = True
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
    def detect(self, imarr: NDArray[np.float64]) -> list[Line2D]:
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
        adrt_result = adrt.adrt(imarr)
        lines = self._find_peaks(adrt_result)
        assert isinstance(lines, list)
        self.log.info(f"The Approximate Discrete Radon Transform detected {len(lines):d} line(s)")
        return lines

    @timed("postprocess")
    def postprocess(
        self,
        streaks: afwTable.SourceTable,
        exposure: afwImage.ExposureF,
        lines: list[Line2D],
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
                streak.setCoord(wcs.pixelToSky(streak_center))

            # Set footprint
            line_mask = get_line_mask(line, shape, self.config.streak_width)
            # Set STREAK mask
            line_masks.append(line_mask)

        self.log.info(f"Accepted {len(streaks):d} streak(s)")

        streak_mask = np.array(line_masks).any(axis=0)
        if self.config.only_mask_detected:
            streak_mask &= detected_mask

        return streak_mask

    def _line_length(self, N: int) -> NDArray[np.float32]:
        """Digital line length of every accumulator cell for domain size ``N``.

        The ADRT of an all-ones image counts the pixels on each digital line,
        which is exactly the number of terms summed into the corresponding
        accumulator cell. Cached per ``N``.
        """
        if N not in self._line_length_cache:
            self._line_length_cache[N] = adrt.adrt(np.ones((N, N), dtype=np.float32))
        return self._line_length_cache[N]

    def _find_peaks(
        self,
        adrt_result: NDArray[np.floating],
        *,
        line_length: NDArray[np.floating] | None = None,
        k: float = 6.0,
        k_contrast: float = 5.0,
        footprint: tuple[int, int] = (7, 31),
        contrast_annulus: tuple[int, int] = (12, 40),
        wing_margin: float = 1.0,
        wing_noise: float = 4.0,
        cone_pad: int = 4,
        min_length: float = 64.0,
        max_peaks: int = 50,
        dedup_rho: float = 3.0,
        dedup_theta: float = 0.5,
        refine: str = "parabolic",
        return_peaks: bool = False,
    ) -> list[Line2D] | tuple[list[Line2D], list[AdrtPeak]]:
        """Detect multiple line peaks in the ADRT accumulator.

        Operates on the significance array ``S = A / sqrt(L)``, where ``A`` is
        the accumulator and ``L`` the digital line length of each cell, so a
        single threshold in noise-sigma units applies to every slope and
        offset. Detection runs per quadrant in four vectorized stages:

        1. Local maxima of ``S`` within an anisotropic ``footprint`` that exceed
           the quadrant's ``median + k * MAD`` and have ``L >= min_length``.
        2. A height-axis local-contrast test that rejects the broad "butterfly
           wing" ridges a bright streak casts across neighbouring slope
           columns: the peak must exceed the median of its own column at
           ``contrast_annulus`` cells away by ``k_contrast * MAD``.
        3. Greedy non-maximum suppression in descending ``S``: once a peak is
           accepted, weaker candidates inside its wing cone
           ``|dh| <= |ds| + cone_pad`` (evaluated in the accepted peak's
           quadrant frame, so the cone reaches across quadrant seams) and
           below the wing envelope are discarded. The envelope is the
           expected wing amplitude of a streak whose ridge has angular
           half-width ``dW``: ``sin(dW) / sin(d) * sqrt(L_peak / L)`` at
           angular distance ``d``, scaled by ``wing_margin``.
        4. Optional sub-cell refinement, closed-form conversion to Hesse
           ``(rho, theta)`` via `_adrt_to_hesse`, and de-duplication in Hesse
           space of lines that appear on both sides of a quadrant seam.

        All coordinates are in the PIXEL frame of the grid the ADRT ran on
        (binned/padded); the caller undoes any binning.

        Parameters
        ----------
        adrt_result : `numpy.ndarray`, (4, 2N-1, N)
            The ADRT accumulator.
        line_length : `numpy.ndarray`, (4, 2N-1, N), optional
            Digital line length (number of valid pixels) per cell. Defaults
            to the ADRT of an all-ones image; pass the ADRT of the valid-pixel
            mask to account for masked regions.
        k : `float`, optional
            Detection threshold above the per-quadrant median, in robust
            (MAD-based) sigma units.
        k_contrast : `float`, optional
            Minimum local contrast over the column background, in the same
            sigma units.
        footprint : `tuple` [`int`, `int`], optional
            Size of the local-maximum neighbourhood along ``(height, slope)``.
        contrast_annulus : `tuple` [`int`, `int`], optional
            Half-open range of height offsets ``[inner, outer)`` on each side of
            a candidate over which the column background is measured.
        wing_margin : `float`, optional
            Multiplier on the wing envelope. A streak of width ``w`` and
            length ``l`` has ridge angular half-width ``dW ~ w / l`` and casts
            a wing of relative accumulator amplitude ``sin(dW) / sin(d)`` at
            angular distance ``d`` (each line at that angle crosses the streak
            over ``w / sin(d)`` pixels); the ``sqrt(L)`` normalization adds the
            line-length factor. Measured wings reach 0.5-0.9 of this model, so
            1.0 suppresses them with a small safety margin. Fainter lines
            crossing inside the cone below the envelope are indistinguishable
            from wings by amplitude and are suppressed too.
        wing_noise : `float`, optional
            Noise allowance added to the wing envelope, in the quadrant's
            MAD-based sigma units. Wing residuals near the detection threshold
            are local maxima of wing plus noise, so they overshoot the
            noise-free envelope by a few sigma; the maximum over the many
            cells of a wing crest reaches about 4 sigma.
        cone_pad : `int`, optional
            Extra height cells added to the wing cone half-width.
        min_length : `float`, optional
            Minimum digital line length for a cell to be a candidate.
        max_peaks : `int`, optional
            Maximum number of peaks accepted by the suppression stage.
        dedup_rho, dedup_theta : `float`, optional
            Two accepted lines closer than this in rho (pixels) and theta
            (degrees) are the same line; the more significant one is kept.
        refine : `str`, optional
            Sub-cell refinement: ``"none"``, ``"parabolic"`` (three-point
            parabola along each axis) or ``"butterfly"``
            (`extract_segment_adrt`, falling back to parabolic where its slope
            band is under-determined). The butterfly moments span the full
            height axis of every slope column, so at unit per-pixel noise the
            noise second moment dominates and the result is unreliable; use it
            only on high-SNR or background-suppressed accumulators.
        return_peaks : `bool`, optional
            Also return the accumulator-space peak descriptors.

        Returns
        -------
        lines : `list` [`~astro_lfd.geom.Line2D`]
            The detected lines, ranked by descending significance.
        peaks : `list` [`AdrtPeak`]
            The matching peak descriptors; only if ``return_peaks`` is set.
        """
        if refine not in ("none", "parabolic", "butterfly"):
            raise ValueError(f"unknown refinement: {refine!r}")
        if adrt_result.ndim != 3 or adrt_result.shape[0] != 4:
            raise ValueError(f"expected an ADRT array of shape (4, 2N-1, N), got {adrt_result.shape}")
        N = adrt_result.shape[2]
        if adrt_result.shape[1] != 2 * N - 1:
            raise ValueError(f"expected an ADRT array of shape (4, 2N-1, N), got {adrt_result.shape}")

        L = self._line_length(N) if line_length is None else np.asarray(line_length)
        S = (adrt_result / np.sqrt(np.maximum(L, 1.0))).astype(np.float32)

        # Stage 1: per-quadrant local maxima above a robust threshold. The
        # filter never crosses the quadrant axis; the noise level is estimated
        # on a 1/64 subsample of the populated cells.
        is_max = S == ndimage.maximum_filter(S, size=(1, *footprint), mode="nearest")
        sub = S[:, ::8, ::8]
        sub_ok = L[:, ::8, ::8] > 0
        cand: list[NDArray[np.intp]] = []
        mads = np.empty(4)
        for q in range(4):
            sample = sub[q][sub_ok[q]]
            median = np.median(sample)
            mads[q] = 1.4826 * np.median(np.abs(sample - median))
            keep = is_max[q] & (S[q] > median + k * mads[q]) & (L[q] >= min_length)
            hs = np.argwhere(keep)
            cand.append(np.column_stack([np.full(len(hs), q), hs]))
        qhs = np.vstack(cand)
        n_local = len(qhs)

        # Stage 2: height-axis local contrast. Wings are nearly as high
        # tens of cells away along h; a real line peak is not.
        if n_local:
            off = np.arange(*contrast_annulus)
            off = np.concatenate([-off, off])
            hh = np.clip(qhs[:, 1, None] + off[None, :], 0, 2 * N - 2)
            background = np.median(S[qhs[:, 0, None], hh, qhs[:, 2, None]], axis=1)
            values = S[qhs[:, 0], qhs[:, 1], qhs[:, 2]]
            keep = (values - background) > k_contrast * mads[qhs[:, 0]]
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
            envelope = np.minimum(1.0, wing_margin * envelope) * values[i] + wing_noise * mads[qhs[:, 0]]
            in_wing = (dh <= ds + cone_pad) & (values <= envelope)
            alive &= ~in_wing

        # Stage 4: refinement, conversion, and seam de-duplication.
        peaks: list[AdrtPeak] = []
        for i in accepted:
            q, h, s = (int(v) for v in qhs[i])
            h_ref, s_ref = float(h), float(s)
            rho_theta: tuple[float, float] | None = None
            if refine == "butterfly":
                try:
                    segment = extract_segment_adrt(adrt_result, q, h, s, N)
                    rho_theta = (segment.rho, segment.theta)
                except ValueError as exc:
                    self.log.debug(f"butterfly refinement failed at {(q, h, s)}: {exc}")
            if rho_theta is None:
                if refine != "none":
                    h_ref = h + _parabolic_offset(S[q, :, s], h)
                    s_ref = s + _parabolic_offset(S[q, h, :], s)
                rho_arr, theta_arr = _adrt_to_hesse(q, h_ref, s_ref, N)
                rho_theta = (float(rho_arr), float(theta_arr))
            rho, theta = rho_theta
            line = Line2D(rho, theta * geom.radians)
            if any(_same_line(line, p.line, dedup_rho, np.deg2rad(dedup_theta)) for p in peaks):
                continue
            peaks.append(AdrtPeak(q, h, s, h_ref, s_ref, float(values[i]), line))

        self.log.debug(
            f"ADRT peaks: {n_local} local maxima, {n_contrast} after contrast test, "
            f"{len(accepted)} after suppression, {len(peaks)} after de-duplication"
        )
        lines = [p.line for p in peaks]
        return (lines, peaks) if return_peaks else lines


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
