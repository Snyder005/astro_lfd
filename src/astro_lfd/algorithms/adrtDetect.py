__all__ = ["ADRTDetectConfig", "ADRTDetectTask"]

import adrt
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from numpy.typing import NDArray

from .base import binary_dilation, get_pixel_mask, timed
from ..geom.line import Line2D
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
        rhos, thetas = self.detect(imarr)
        self.postprocess(streaks, exposure, rhos=rhos, thetas=thetas)

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
    def detect(self, imarr: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Perform linear feature detection on masked image array.

        Currently the peak detector is not fully implemented; it returns the
        global maximum for research and development. Hesse normal form should
        return parameters in the PIXEL frame.

        Parameters
        ----------
        imarr : `numpy.ndarray`
            The masked image array.

        Returns
        -------
        rhos, thetas : `numpy.ndarray`, (N,)
            The Hesse normal form rho and theta parameters of the detected
            lines.
        """
        adrt_result = adrt.adrt(imarr)

        # Peak detector (to be developed fully)
        rhos, thetas = self._find_peaks(adrt_result)

        return rhos * self.config.bin_size, thetas

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

        self.log.info("Accepted %d streak(s) after profile fitting", len(streaks))

    def _adrt_to_hesse(self, q: int, h: int, s: int, N: int) -> tuple[float, float]:
        """Convert ADRT coordinates to Hesse normal form parameters.

        Hesse normal parameters are defined in the PIXEL coordinate system.
        This currently only operates on a single set of ADRT coordinates, but
        it should be vectorized to quickly convert multiple ADRT coordinates.

        Parameters
        ----------
        q, h, s : `int`
            The ADRT result quadrant, slope, and height indices.
        N : `int`
            Size of the ADRT domain (must be a power of 2).

        Returns
        -------
        rho, theta : `float`
            The Hesse normal form rho and theta (in rad) parameters.
        """
        c = (N - 1) / 2.0
        offset, angle = adrt.utils.coord_adrt(N)  # This may be slowest part

        theta = np.pi / 2 - angle[q, 0, s]
        rho_center = offset[q, h, s] * N * -1.0
        rho = rho_center + c * (np.cos(theta) + np.sin(theta))

        return rho, theta

    def _find_peaks(
        self,
        adrt_result: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Placeholder for peak finding in the ADRT transform space.

        Will eventually detect multiple peaks, which are then transformed and
        packaged into a form to send to postprocessing. Focus on
        implementation first, then decide optimizations (within Python or as
        an extension to a branch of `adrt` if C++ implementation needed).

        Parameters
        ----------
        adrt_result : `numpy.ndarray`
            The ADRT result.

        Returns
        -------
        rhos, thetas : `numpy.ndarray`, (N,)
            The Hesse normal form rho and theta (in rad) parameters of the
            detected lines.
        """
        # Get global maximum indices (placeholder for multpeak finding)
        q, h, s = np.unravel_index(np.argmax(adrt_result), adrt_result.shape)

        # Get window around global indices (to be used in ADRT space analysis)
        # Not Implemented Yet

        N = adrt_result.shape[2]
        rho, theta = self._adrt_to_hesse(q, h, s, N)

        return np.array(rho), np.array(theta)
