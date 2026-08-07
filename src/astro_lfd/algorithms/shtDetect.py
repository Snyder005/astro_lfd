from scipy import ndimage
from scipy.stats import median_abs_deviation
import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import pyhough

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
    nsig_det = pexConfig.Field(
        doc="Number of sigma above binned background to search for streaks.",
        dtype=float,
        default=1.0,
    )
    nsig_streak = pexConfig.Field(
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

        det, bad_mask = self.preprocess(exposure)
        rhos, thetas = self.detect(det, bad_mask)
        self.postprocess(streaks, exposure, rhos, thetas)

        return pipeBase.Struct(streaks=streaks, timings=self.timings)

    @timed("preprocess")
    def preprocess():
        bad_mask = get_pixel_mask(mask, list(self.config.bad_mask_planes))

        sky_noise = None
        if (info := exposure.getInfo()) is not None: # check that exposure has method
            if (stats := exposure.getSummaryStats()) is not None: # check that exposure has method
                if np.isfinite(stats.skyNoise):
                    sky_noise = stats.skyNoise

        if sky_noise is None:
            sky_noise = median_abs_deviation(
                exposure.image.array[(exposure.mask.array & bad_mask) == 0].ravel(),
                scale="normal",
            )

        binned = afwMath.binImage(exposure.maskedImage, binning)
        binned_noise = sky_noise / binning

        det = (binned.image.array > (nsig_det * binned_noise))
        det[(binned.mask.array & bad_mask) > 0] = False

        return det, bad_mask

    @timed("detect")
    def detect(det, bad_mask):

        hough = pyhough.Hough(det)
        transform = hough.transform()

        det_template = np.ones_like(det)
        det_template[(binned.mask.array & bad_mask) > 0] = False
        hough_template = pyhough.Hough(det_template)
        transform_template = hough_template.transform()
        ratio = transform[0] / transform_template[0]

        r_use = np.isfinite(ratio)

        med = np.median(ratio[r_use])
        sig = median_abs_deviation(ratio[r_use].ravel(), scale="normal")

        high_bool = (ratio > (med + nsig_streak * sig)) & (transform_template[0] > min_streak_length / binning)

        labeled, n = ndimage.label(high_bool)
        counts = np.bincount(labeled.ravel())
        keep = np.where(counts[1:] >= min_cluster_size)[0] + 1

        thetas = np.zeros(len(keep))
        rhos = np.zeros(len(keep))

        for i, label in enumerate(keep):
            ys, xs = np.where(labeled == label)

            thetas[i] = np.rad2deg(np.median(transform[1][xs]))
            rhos[i] = np.median(transform[2][ys]) * binning

        return rhos, thetas

    @timed("postprocess")
    def postprocess(self, streaks, exposure, rhos, thetas):
        box = exposure.getBBox()
        shift = geom.Extent2D(box.getCenter())
        for rho, theta in np.nditer((rhos, thetas)):
            line = Line2D(rho, theta * geom.degrees)
            line_segment = line.intersection(box)
            if line_segment is None:
                continue
            center = line_segment.center

#            footprint_mask = afwImage.Mask(final_line_mask.astype(np.int32))
#            spans = afwGeom.SpanSet.fromMask(footprint_mask)
#            footprint = afwDetect.Footprint(spans)

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(line_segment)
            if (wcs := exposure.getWcs()) is not None:
                streak.setCoord(wcs.pixelToSky(center))
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()

        self.log.info("Accepted %d streak(s) after profile fitting", len(streaks))
