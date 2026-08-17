__all__ = ["KHTDetectConfig", "KHTDetectTask"]

import math
import numpy as np
from numpy.typing import NDArray
from skimage.feature import canny
from sklearn.cluster import KMeans

import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.kht
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms.maskStreaks import Line, LineCollection, LineProfile

from .base import binary_dilation, get_pixel_mask, timed
from ..geom.line import Line2D
from ..table.streakAdapter import StreakAdapter


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
    bin_size = pexConfig.Field(
        doc="Pixel bin size for input image.",
        dtype=int,
        default=1,
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
    """Detect straight linear features with the Kernel Hough Transform."""

    ConfigClass = KHTDetectConfig
    _DefaultName = "khtDetect"

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
            ``edges``
                Canny binary edge map with invalid regions masked
                (`numpy.ndarray`).
            ``timings``
                Computing times for each processing step (`dict`)
        """
        streaks = afwTable.SourceCatalog(table)
        self.timings = {}
        edges = self.preprocess(exposure)
        rhos, thetas = self.detect(edges)
        self.postprocess(streaks, exposure, rhos=rhos, thetas=thetas)

        return pipeBase.Struct(streaks=streaks, edges=edges, timings=self.timings)

    @timed("preprocess")
    def preprocess(
        self,
        exposure: afwImage.ExposureF,
    ) -> NDArray[np.bool_]:
        """Perform preprocessing on input exposure.

        Parameters
        ----------
        exposure: `lsst.afw.image.ExposureF`
            The exposure to search. The mask plane named by
            ``config.detected_mask_plane`` must flag the detected pixels.

        Returns
        -------
        edges : `numpy.ndarray`, (Ny, Nx)
            The Canny binary edge map with invalid regions masked.
        """
        mi = afwMath.binImage(exposure.maskedImage, self.config.bin_size)
        detected_mask = get_pixel_mask(mi.mask, self.config.detected_mask_plane)
        bad_mask = get_pixel_mask(mi.mask, self.config.bad_mask_planes)
        init_edges = canny(detected_mask.astype(np.float64), use_quantiles=True, sigma=0.1)

        dilated_bad_mask = binary_dilation(bad_mask, 1) if self.config.bin_size == 1 else bad_mask
        if self.config.saturated_detections_dilation:
            sat_mask = get_pixel_mask(mi.mask, "SAT")
            sat_detected_mask = binary_dilation(
                sat_mask & detected_mask,
                math.floor(self.config.saturated_detections_dilation / self.config.bin_size),
            )
            invalid = sat_detected_mask | dilated_bad_mask
        else:
            invalid = dilated_bad_mask

        edges = init_edges & ~invalid
        return edges

    @timed("detect")
    def detect(self, edges: NDArray[np.bool_]) -> tuple(NDArray[np.float64], NDArray[np.float64]):
        """Perform linear feature detection on Canny binary edge map.

        Parameters
        ----------
        edges : `numpy.ndarray`, (Ny, Nx)
            The Canny binary edge map.

        Returns
        -------
        rhos, thetas : `numpy.ndarray`
            The Hesse normal form rho and theta parameters of the detected
            lines.
        """
        lines = lsst.kht.find_lines(
            edges,
            math.floor(self.config.cluster_minimum_size / self.config.bin_size),
            self.config.cluster_minimum_deviation / self.config.bin_size,
            self.config.delta,
            self.config.minimum_kernel_height,
            self.config.nsigma,
            self.config.abs_minimum_kernel_height / self.config.bin_size**2,
        )

        self.log.info("The Kernel Hough Transform detected %d line(s)", lines.size)
        return lines.rho * self.config.bin_size, lines.theta

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

        Parameters
        ----------
        streaks : `lsst.afw.table.SourceTable`
            The output streak catalog.
        exposure : `lsst.afw.image.ExposureF`
            The exposure that was searched.
        rhos, thetas : `numpy.ndarray`
            The Hesse normal form rho and theta parameters of the detected
            lines after cluster consolidation.
        """
        if rhos.size == 0:
            return

        rhos, thetas = self._cluster_lines(rhos, thetas)
        bad_mask = get_pixel_mask(exposure.mask, self.config.bad_mask_planes)
        detected_mask = get_pixel_mask(exposure.mask, self.config.detected_mask_plane)
        weights = exposure.variance.array**-1
        weights[~np.isfinite(weights) | ~np.isfinite(exposure.image.array)] = 0
        weights[bad_mask] = 0

        box = exposure.getBBox()
        wcs = exposure.getWcs()
        shift = geom.Extent2D(box.getCenter())
        for rho, theta in np.nditer((rhos, thetas)):
            line = Line(float(rho), float(theta), sigma=self.config.inv_sigma**-1)
            line_model = LineProfile(exposure.image.array, weights, line=line, detectionMask=detected_mask)
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
            streak["line_sigma"] = fit.sigma
            streak["line_reduced_chi2"] = fit.reducedChi2
            streak["line_model_maximum"] = float(model_maximum)

        self.log.info("Accepted %d streak(s) after profile fitting", len(streaks))

    def _cluster_lines(
        self,
        rhos: NDArray[np.float64],
        thetas: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cluster nearby lines by recursive k-means clustering.

        Parameters
        ----------
        rhos, thetas : `numpy.ndarray`
            The Hesse normal form rho and theta parameters of the detected
            lines.

        Returns
        -------
        rhos, thetas : `numpy.ndarray`
            The Hesse normal form rho and theta parameters of the consolidated
            cluster centers.
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
