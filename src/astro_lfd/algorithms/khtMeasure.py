__all__ = ["KHTMeasureConfig", "KHTMeasureTask"]

import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np

from ..geom.line import Line2D


class KHTMeasureConfig(pexConfig.Config):
    """Configurable parameters for `KHTMeasureTask`."""

    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak measurement.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    detected_mask_plane = pexConfig.Field(
        doc="Name of mask plane with pixels above detection threshold.",
        dtype=str,
        default="DETECTED",
    )
    rho_bin_size = pexConfig.Field(
        doc="Binsize (in pixels) for position parameter when finding clusters.",
        dtype=float,
        default=40.0,
    )
    theta_bin_size = pexConfig.Field(
        doc="Binsize (in degrees) for angle parameter when finding clusters.",
        dtype=float,
        default=2.0,
    )
    inv_sigma = pexConfig.Field(
        doc="Inverse of the Moffat sigma parameter (in pixels) describing the streak profile.",
        dtype=float,
        default=10.0**-1,
    )
    dchi2_tolerance = pexConfig.Field(
        doc="Absolute difference in chi2 between fit iterations for convergence.",
        dtype=float,
        default=0.1,
    )
    max_fit_iter = pexConfig.Field(
        doc="Maximum number of fit iterations acceptable for convergence.",
        dtype=int,
        default=100,
    )
    max_streak_width = pexConfig.Field(
        doc="Maximum width (in pixels) of the streak mask.",
        dtype=float,
        default=0.0,
    )

class KHTMeasureTask(pexConfig.Config):
    """Measure stuff."""

    ConfigClass = KHTMeasureConfig
    _DefaultName = "khtMeasure"

    def run(self, table: afwTable.SourceCatalog, lines: list[Line2D], exposure: afwImage.ExposureF):

        data = exposure.image.array # sent to lineprofile
        weights = exposure.variance.array**-1 # sent to lineprofile
        weights[~np.isfinite(weights) | ~np.isfinite(data)] = 0
        weights[get_pixel_mask(exposure.mask, self.config.bad_mask_planes)] = 0
        detect_mask = get_pixel_mask(exposure.mask, self.config.detect_mask_plane) # sent to lineprofile

        for line in lines:
            sigma = self.config.inv_sigma**-1
            line_model = LineProfile(data, weights, line=line, detectionMask=detectionMask)
            if line_model.modelFailure or line_model.lineMask.sum() == 0:
                continue

            fit, failure = line_model.fit(self.config.dchi2_tolerance, maxIter=self.config.max_fit_iter)

            if (abs(fit.rho - line.rho) > 2 * self.config.rho_bin_size) or (
                abs(fit.theta - line.theta) > 2 * self.config.theta_bin_size
            ):
                failure = True

            if failure:
                continue

            line_model.setLineMask(fit, self.config.max_streak_width, self.config.nsigma_mask)
            final_model = line_model.makeProfile(fit)
            model_maximum = abs(final_model).max()
            final_line_mask = abs(final_model) > self.config.footprint_threshold
            if not final_line_mask.any():
                continue

            kht_line = Line2D(fit.rho, fit.theta * geom.degrees)
            line = kht_line.translated(shift)
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
            streak["line_sigma"] = fit.sigma
            streak["line_reduced_chi2"] = fit.reducedChi2
            streak["line_model_maximum"] = float(model_maximum)

            # Individual streak mask
            line_mask = get_line_mask(line, shape, self.config.streak_width)
            line_masks.append(line_mask)
        self.log.info(f"Accepted {len(streaks):d} streak(s)")

        streak_mask = np.array(line_masks).any(axis=0)
        if self.config.only_mask_detected:
            streak_mask &= get_pixel_mask(exposure.mask, self.config.detected_mask_plane)
        return lineFits, finalMask
