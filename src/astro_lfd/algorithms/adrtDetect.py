__all__ = ["ADRTDetectConfig", "ADRTDetectTask"]

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from numpy.typing import NDArray

from .base import binary_dilation, get_pixel_mask, timed

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

    # Configuration for detection

    # Configuration for postprocessing


class ADRTDetectTask(pipeBase.Task):
    """Detect linear features with the Approximate Discrete Radon Transform.
    """

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
            ``timings``
                Computing times for each processing step (`dict`)
        """
        streaks = afwTable.SourceCatalog(table)
        self.timings = {}
        imarr = self.preprocess(exposure)
        rhos, thetas = self.detect(imarr)
        self.postprocess(streaks, exposure, rhos=rhos, thetas=thetas)

        return pipeBase.Struct(streaks=streaks, timings=self.timings)

    @timed("preprocess")
    def preprocess(self, exposure: afwImage.ExposureF) -> NDArray[np.float64]:
        """Perform preprocessing on input exposure.

        Parameters
        ----------
        exposure: `lsst.afw.image.ExposureF`
            The exposure to search.

        Returns
        -------
        imarr : `numpy.ndarray`, (Ny, Nx)
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
            ((0, 4096 // config_bin_size - imarr.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        return padded_imarr

    @timed("detect")
    def detect(self, imarr: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Perform linear feature detection on masked image array.

        Currently the peak detector is not fully implemented; it returns the
        global maximum for research and development. Hesse normal form should
        return parameters in the PIXEL frame.

        Parameters
        ----------
        imarr : `numpy.ndarray`, (Ny, Nx)
            The masked image array.

        Returns
        -------
        rhos, thetas : `numpy.ndarray`
            The Hesse normal form rho and theta parameters of the detected
            lines.
        """
        adrt_result = adrt.adrt(imarr)

        # Peak detector (to be developed fully)
        peaks = self._find_peaks(adrt_result)

        return peaks.rho, peaks.theta

    @timed("postprocess")
    def postprocess(
        self,
        streaks: afwTable.SourceTable,
        exposure: afwImage.ExposureF,
        *,
        rhos: NDArray[np.float64],
        thetas: NDArray[np.float64],
    ) -> None:
        """Perform postprocessing of detected linear features.

        Currently this only adds the line segment representation of each
        detected streak to the streak catalog. For ADRT assume Hesse normal
        origin is PIXEL origin, so no translation is needed.

        Parameters
        ----------
        streaks : `lsst.afw.table.SourceTable`
            The output streak catalog.
        exposure : `lsst.afw.image.ExposureF`
            The exposure that was searched.
        rhos, thetas : `numpy.ndarray`, (N,)
            The Hesse normal form rho and theta parameters of the detected
            lines.
        """
        if rhos.size == 0:
            return

        box = exposure.getBBox()
        wcs = exposure.getWcs()
        for rho, theta in np.nditer((rhos, thetas)):
            line = Line2D(rho, theta * geom.radians)
            line_segment = line.intersection(box)
            if line_segment is None:
                continue

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(line_segment)

            center = line_segment.center
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()
            if wcs is not None:
                streak.setCoord(wcs.pixelToSky(center))

        self.log.info("Accepted %d streak(s) after profile fitting", len(streaks))

    def _find_peaks(self, adrt_result: NDArray[np.float64]):
        """Placeholder for peak finding in the ADRT transform space.

        Will eventually detect multiple peaks, which are then transformed and
        packaged into a form to send to postprocessing.

        See https://adrt.readthedocs.io/en/latest/examples.coordinate.html for
        reference.

        Parameters
        ----------
        adrt_result : `numpy.ndarray`
            Result of ADRT transformation.

        Returns
        -------
        rhos, thetas : `numpy.ndarray`, (N,)
            The Hesse normal form rho and theta parameters of the detected
            lines.
        """
        # Get global maximum indices
        q, h, s = np.unravel_index(np.argmax(adrt_result), adrt_result.shape)

        # Given the image dimensions N x N, get the offset and angle coordinate arrays 
        # for the corresponding Radon transform space
        N = adrt_result.shape[2] 
        offset, angle = adrt.utils.coord_adrt(N)

        # Proposed conversions 
        # 1. Get offset corresponding to the peak coordinates (q, h, s)
        # 2. Scale by N (see https://adrt.readthedocs.io/en/latest/examples.coordinate.html)
        # 3. Multiply by -1 because geometric orientation is default, not origin="lower"
        rho = offset[q, h, s] * N * -1.0
        
        # 1. Get theta correspond to the peak coordinates (q, 0, s)
        # Note: This only depends on quadrant and slope, not height
        # 2. Derive mapping from the angle (measured as between x-axis and the 
        # line, NOT typical Hough). This seems to be pi/2 minus the angle (but
        # have not tried all cases).
        theta = np.pi / 2 - angle[q, 0, s] # in radians

        # Return should be rho (in pixels) and theta (in radians) in Hesse
        # normal, but with origin at image center:
        # - This is NOT the exposure bounding box center 
        # (`exposure.getBBox().getCenter()`) because the exposure has been
        # padded.
        # - The ADRT result, including the offset/angle coordinate arrays are
        # for a potentially binned image.
        # This is not in the correct form yet (PIXEL coordinate system origin).
        return np.array(rho), np.array(theta)
