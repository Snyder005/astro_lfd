__all__ = ["ADRTDetectConfig", "ADRTDetectTask", "ADRTSegment", "extract_segment_adrt"]

import adrt
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from numpy.typing import NDArray

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
        bad_mask = get_pixel_mask(mi.mask, self.config.bad_mask_planes, dilation=1)
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
        # Peak detector (to be developed fully). Returns integer accumulator
        # indices in the binned/padded grid the ADRT actually ran on.
        result = self._find_peaks(adrt_result)
        self.log.info(f"The Approximate Discrete Radon Transform detected {result.size:d} line(s)")
        return [] if result.size == 0 else result

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
