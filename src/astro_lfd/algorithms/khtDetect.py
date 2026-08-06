"""Kernel Hough Transform (KHT) streak detection as an LSST task.

`KHTDetectTask` finds straight linear features (satellite streaks, and similar
signals) in an `lsst.afw.image.Exposure`.  It reproduces the detection stages of
`lsst.meas.algorithms.maskStreaks.MaskStreaksTask` -- Canny edge extraction,
`lsst.kht` line finding, recursive k-means clustering, and a Moffat line-profile
fit -- but emits its results as a `~lsst.afw.table.SourceCatalog` of canonical
line segments (via `~astro_lfd.table.streakAdapter.StreakAdapter`) instead of a
mask plane.

The profile fitter (`Line`, `LineProfile`) is imported from ``maskStreaks``
rather than reimplemented, so any difference in the fit itself is shared between
the two tasks.  This module is intended as the template for future
``astro_lfd`` detector tasks, including the ADRT-based detector.
"""

# mypy: disable-error-code="var-annotated, attr-defined"
# `lsst.pex.config.Field` descriptors are declared in the `Config` class body
# without a type annotation and read back as instance attributes; mypy cannot
# model this dynamic descriptor protocol, so the resulting false positives are
# silenced for this module.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny
from sklearn.cluster import KMeans

import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.kht
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms.maskStreaks import Line, LineProfile
from lsst.utils.timer import timeMethod

from ..geom.line import Line2D
from ..table.streakAdapter import StreakAdapter

__all__ = ["KHTDetectConfig", "KHTDetectTask", "get_pixel_mask", "binary_dilation"]


def get_pixel_mask(mask: afwImage.Mask, mask_plane: str | list[str]) -> NDArray[np.bool_]:
    """Get the pixel mask array corresponding to the named mask planes.

    Parameters
    ----------
    mask : `lsst.afw.image.Mask`
        The input mask.
    mask_plane : `str` or `list` [`str`]
        Name or list of names of the mask plane(s).

    Returns
    -------
    pixel_mask : `numpy.ndarray`, (Ny, Nx)
        Boolean array, `True` where any of the named planes is set.
    """
    return (mask.array & mask.getPlaneBitMask(mask_plane)) != 0


def binary_dilation(binary_image: NDArray[np.bool_], npix_to_dilate: int) -> NDArray[np.bool_]:
    """Dilate a binary array with a circular structuring element.

    Parameters
    ----------
    binary_image : `numpy.ndarray`, (Ny, Nx)
        The input binary image array.
    npix_to_dilate : `int`
        Pixel radius of the circular structuring element to dilate by.

    Returns
    -------
    dilated_image : `numpy.ndarray`, (Ny, Nx)
        The dilated binary image array.
    """
    return distance_transform_edt(~binary_image) <= npix_to_dilate


class KHTDetectConfig(pexConfig.Config):
    """Configurable parameters for `KHTDetectTask`."""

    # Configuration for pixel masking.
    detected_mask_plane = pexConfig.Field(
        doc="Name of mask plane with pixels above detection threshold.",
        dtype=str,
        default="DETECTED",
    )
    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    saturated_detections_dilation = pexConfig.Field(
        doc="Number of pixels to dilate the saturated detections mask.",
        dtype=int,
        default=250,
    )

    # Configuration for KHT.
    cluster_minimum_size = pexConfig.Field(
        doc="Minimum size (in pixels) of detected clusters.",
        dtype=int,
        default=50,
    )
    cluster_minimum_deviation = pexConfig.Field(
        doc="Allowed deviation (in pixels) from a straight line.",
        dtype=int,
        default=2,
    )
    delta = pexConfig.Field(
        doc="Stepsize in angle-radius parameter space.",
        dtype=float,
        default=0.2,
    )
    minimum_kernel_height = pexConfig.Field(
        doc="Minimum height of the streak-finding kernel relative to the tallest kernel.",
        dtype=float,
        default=0.0,
    )
    nsigma = pexConfig.Field(
        doc="Number of sigma from center of kernel to include in voting procedure.",
        dtype=float,
        default=2.0,
    )
    abs_minimum_kernel_height = pexConfig.Field(
        doc="Minimum absolute height of the streak-finding kernel.",
        dtype=float,
        default=5.0,
    )

    # Configuration for clustering.
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

    # Configuration for profile fit.
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
    nsigma_mask = pexConfig.Field(
        doc="Number of sigma from center of kernel to mask.",
        dtype=float,
        default=5.0,
    )
    footprint_threshold = pexConfig.Field(
        doc="Threshold at which to determine the edge of a line (in nanoJansky).",
        dtype=float,
        default=0.01,
    )


class KHTDetectTask(pipeBase.Task):
    """Detect straight linear features with the Kernel Hough Transform.

    The task runs the same detection stages as
    `lsst.meas.algorithms.maskStreaks.MaskStreaksTask` (Canny edges, `lsst.kht`
    line finding, recursive k-means clustering, and a Moffat profile fit) but
    records each accepted line as a `~astro_lfd.table.streakAdapter.StreakAdapter`
    row in a `~lsst.afw.table.SourceCatalog`.
    """

    ConfigClass = KHTDetectConfig
    _DefaultName = "khtDetect"

    @timeMethod
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
        image = exposure.image.array
        variance = exposure.variance.array
        mask = exposure.mask
        streaks = afwTable.SourceCatalog(table)
        wcs = exposure.getWcs()

        # Calculate the Canny edge response of the detected pixels.
        detected_mask = get_pixel_mask(mask, self.config.detected_mask_plane)
        init_edges = canny(detected_mask.astype(np.float64), use_quantiles=True, sigma=0.1)

        # Mask invalid regions before line finding.  The bad planes get a
        # one-pixel buffer for the edge image (so the borders of bad regions
        # are also ignored), but the fit weights below are zeroed on the
        # *undilated* bad planes to match `MaskStreaksTask._fitProfile`.
        bad_mask = get_pixel_mask(mask, list(self.config.bad_mask_planes))
        dilated_bad_mask = binary_dilation(bad_mask, 1)
        if self.config.saturated_detections_dilation:
            sat_mask = get_pixel_mask(mask, "SAT")
            sat_detected_mask = binary_dilation(
                sat_mask & detected_mask,
                self.config.saturated_detections_dilation,
            )
            invalid = sat_detected_mask | dilated_bad_mask
        else:
            invalid = dilated_bad_mask
        edges = init_edges & ~invalid

        # Run the Kernel Hough Transform on the edge image.
        lines = lsst.kht.find_lines(
            edges,
            self.config.cluster_minimum_size,
            self.config.cluster_minimum_deviation,
            self.config.delta,
            self.config.minimum_kernel_height,
            self.config.nsigma,
            self.config.abs_minimum_kernel_height,
        )
        self.log.info("The Kernel Hough Transform detected %d line(s)", lines.size)
        if lines.size == 0:
            return pipeBase.Struct(streaks=streaks, edges=edges)

        rhos, thetas = self._cluster_lines(lines.rho, lines.theta)

        # Build fit weights: inverse variance, with invalid pixels zeroed.
        # Zero on the undilated bad planes (not the one-pixel-dilated edge
        # mask) so the pixels driving the profile fit match those used by
        # `MaskStreaksTask._fitProfile`.
        weights = variance**-1
        weights[~np.isfinite(weights) | ~np.isfinite(image)] = 0
        weights[bad_mask] = 0

        # The profile fit works in image-array coordinates centered on the
        # image, so map the fit line into absolute pixel coordinates by
        # translating from the image center. Using the exposure bounding box
        # keeps the frame consistent with the fitted image array and avoids a
        # hard dependence on the detector being attached.
        box = exposure.getBBox()
        shift = geom.Extent2D(box.getCenter())
        for rho, theta in np.nditer((rhos, thetas)):
            line = Line(float(rho), float(theta), sigma=self.config.inv_sigma**-1)
            line_model = LineProfile(image, weights, line=line, detectionMask=detected_mask)
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
            det_line = kht_line.translated(shift)
            line_segment = det_line.intersection(box)
            if line_segment is None:
                continue
            center = line_segment.center

            footprint_mask = afwImage.Mask(final_line_mask.astype(np.int32))
            spans = afwGeom.SpanSet.fromMask(footprint_mask)
            footprint = afwDetect.Footprint(spans)

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(line_segment)
            streak.setFootprint(footprint)
            if wcs is not None:
                streak.setCoord(wcs.pixelToSky(center))
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()
            # Carry the profile-fit quality metrics.  `fit` is a maskStreaks
            # `Line` whose `reducedChi2` field holds the reduced chi-squared.
            streak["line_sigma"] = fit.sigma
            streak["line_reduced_chi2"] = fit.reducedChi2
            streak["line_model_maximum"] = float(model_maximum)

        self.log.info("Accepted %d streak(s) after profile fitting", len(streaks))
        return pipeBase.Struct(streaks=streaks, edges=edges)

    def _cluster_lines(
        self,
        rhos: NDArray[np.float64],
        thetas: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cluster nearby lines by recursive k-means clustering.

        The rho/theta parameters are rescaled by their configured bin sizes so
        that both axes are comparable, then k-means is run with an increasing
        number of clusters until every cluster has a per-axis standard deviation
        at or below one bin.  The cluster centers are the consolidated lines.

        Parameters
        ----------
        rhos, thetas : `numpy.ndarray`
            The rho (pixels) and theta (degrees) parameters of the detected
            lines.

        Returns
        -------
        rhos, thetas : `numpy.ndarray`
            The rho and theta parameters of the consolidated cluster centers.
        """
        x = rhos / self.config.rho_bin_size
        y = thetas / self.config.theta_bin_size
        points = np.column_stack((x, y))

        n_clusters = 1
        while True:
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(points)
            cluster_standard_deviations = np.zeros((n_clusters, 2))
            for c in range(n_clusters):
                in_cluster = points[kmeans.labels_ == c]
                cluster_standard_deviations[c] = np.std(in_cluster, axis=0)

            if (cluster_standard_deviations <= 1).all():
                break

            n_clusters += 1

        final_clusters = kmeans.cluster_centers_.T
        rhos = final_clusters[0] * self.config.rho_bin_size
        thetas = final_clusters[1] * self.config.theta_bin_size

        return rhos, thetas
