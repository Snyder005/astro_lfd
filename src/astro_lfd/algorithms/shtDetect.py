from numpy.typing import NDArray
from scipy import ndimage
from scipy.stats import median_abs_deviation
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import warnings

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from .utils import get_pixel_mask, timed
from ..geom.line import Line2D
from ..table.streakAdapter import StreakAdapter

__all__ = ["SHTDetectConfig", "SHTDetectTask"]


class SHTDetectConfig(pexConfig.Config):
    """Configurable parameters for `SHTDetectTask`."""

    # Configuration for pixel masking
    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )

    binning = pexConfig.Field(
        doc="Bin factor for finding.",
        dtype=int,
        default=16,
    )

    # Configuration for SHT
    nsigma = pexConfig.Field(
        doc="Number of sigma above binned background to search for streaks.",
        dtype=float,
        default=1.0,
    )
    nsigma_streak = pexConfig.Field(
        doc="Number of sigma to record the streak.",
        dtype=float,
        default=5.0,
    )
    min_cluster_size = pexConfig.Field(
        doc="Minimum size of detected clusters.",
        dtype=int,
        default=5,
    )
    min_streak_length = pexConfig.Field(
        doc="Minimum streak length (edge to edge) to consider.",
        dtype=int,
        default=1500,
    )

    
class SHTDetectTask(pipeBase.Task):
    """Detect straight linear features with the Simple Hough Transform."""

    ConfigClass = SHTDetectConfig
    _DefaultName = "shtDetect"

    timings: dict[str, float]
    
    def run(self, table: afwTable.SourceTable, exposure: afwImage.ExposureF) -> pipeBase.Struct:
        """Detect streaks in an exposure.

        Parameters
        ----------
        table : `lsst.afw.table.SourceTable`
            Source table used to construct the output catalog.  Its schema must
            provide the streak ``line_*`` fields (see
            `~astro_lfd.table.streakAdapter.StreakAdapter.makeMinimalSchema`).
        exposure : `lsst.afw.image.ExposureF`
            Exposure to search.  The mask plane named by
            ``config.detected_mask_plane`` must flag the detected pixels.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result as a struct with attributes:

            ``streaks``
                Catalog of detected streaks (`lsst.afw.table.SourceCatalog`).
            ``edges``
                Boolean Canny edge image used for line finding, with invalid 
                regions removed (`numpy.ndarray`).
        """
        streaks = afwTable.SourceCatalog(table)

        self.timings = {}
        detected_mask, bad_mask = self.preprocess(exposure)
        rhos, thetas = self.detect(detected_mask, bad_mask)
        self.postprocess(streaks, exposure, rhos=rhos, thetas=thetas)

        return pipeBase.Struct(streaks=streaks, timings=self.timings)

    @timed("preprocess")
    def preprocess(self, exposure: afwImage.ExposureF) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        mi = afwMath.binImage(exposure.maskedImage, self.config.binning)
        bad_mask = get_pixel_mask(mi.mask, self.config.bad_mask_planes)

        sky_noise = None
        if (info := exposure.getInfo()) is not None:
            if (stats := info.getSummaryStats()) is not None:
                if np.isfinite(stats.skyNoise):
                    sky_noise = stats.skyNoise

        if sky_noise is not None:
            detection_noise = sky_noise / self.config.binning
        else:
            detection_noise = median_abs_deviation(
                mi.image.array[~bad_mask].ravel(),
                scale="normal",
            )

        detected_mask = mi.image.array > (self.config.nsigma * detection_noise)
        detected_mask[bad_mask] = False

        return detected_mask, bad_mask

    @timed("detect")
    def detect(
        self,
        detected_mask: NDArray[np.bool_],
        bad_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        tested_angles = np.linspace(0.0, np.pi, 360, endpoint=False)
        hspace, angles, distances = hough_line(detected_mask, theta=tested_angles)

        detected_mask_template = ~bad_mask
        hspace_template, _, _ = hough_line(detected_mask_template, theta=tested_angles)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio = hspace / hspace_template

        ratio[~np.isfinite(ratio)] = 0.0
        peaks, thetas, rhos = hough_line_peaks(ratio, angles, distances)

        return rhos, thetas

    @timed("postprocess")
    def postprocess(
        self,
        streaks: afwTable.SourceTable,
        exposure: afwImage.ExposureF,
        *,
        rhos: NDArray[np.float64],
        thetas: NDArray[np.float64],
    ):

        box = exposure.getBBox()
        wcs = exposure.getWcs()
        print(rhos, thetas)
        for rho, theta in np.nditer((rhos, thetas)):
            line = Line2D(rho * self.config.binning, theta * geom.radians)
            line_segment = line.intersection(box)
            if line_segment is None:
                continue

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(line_segment)

            center = line_segment.center
            if wcs is not None:
                streak.setCoord(wcs.pixelToSky(center))
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()

        self.log.info("Accepted %d streak(s)", len(streaks))
