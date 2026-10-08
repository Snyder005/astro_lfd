__all__ = ["KHTDetectConfig", "KHTDetectTask"]

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.kht
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from numpy.typing import NDArray
from skimage.feature import canny
from sklearn.cluster import KMeans

from .base import get_line_mask, get_pixel_mask, timed
from ..geom.line import Line2D
from ..table.streakAdapter import StreakAdapter


class KHTDetectConfig(pexConfig.Config):
    """Configurable parameters for `KHTDetectTask`."""

    # Configuration for preprocess.
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
    bad_mask_dilation = pexConfig.Field(
        doc="Number of pixels to dilate the bad mask by.",
        dtype=int,
        default=10,
    )

    # Configuration for KHT.
    cluster_minimum_size = pexConfig.Field(
        doc="Minimum size (in pixels) of detected clusters.",
        dtype=int,
        default=50,
    )
    cluster_minimum_deviation = pexConfig.Field(
        doc="Allowed deviation (in pixels) from a straight line for a detected feature.",
        dtype=int,
        default=2,
    )
    delta = pexConfig.Field(
        doc="Step size in angle-radius parameter space.",
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

    # Configuration for postprocess
    only_mask_detected = pexConfig.Field(
        doc="If true, only propagate the part of the streak mask that overlaps with the detection mask.",
        dtype=bool,
        default=True,
    )
    streak_width = pexConfig.Field(
        doc="Initial width (in pixels) of the streak mask.",
        dtype=float,
        default=200.0,
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
            ``streak_mask``
                Streak mask plane (`numpy.ndarray`).
            ``timings``
                Computing times for each processing step (`dict`)
        """
        streaks = afwTable.SourceCatalog(table)
        self.timings = {}

        edges, bad_mask = self.preprocess(exposure)
        lines = self.detect(edges)
        streak_mask = self.postprocess(streaks, exposure, lines)

        return pipeBase.Struct(
            streaks=streaks,
            edges=edges,
            bad_mask=bad_mask,
            streak_mask=streak_mask,
            timings=self.timings,
        )

    @timed("preprocess")
    def preprocess(self, exposure: afwImage.ExposureF) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
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
        detected_mask = get_pixel_mask(exposure.mask, self.config.detected_mask_plane)
        edges = canny(detected_mask.astype(np.float64), use_quantiles=True, sigma=0.1)
        bad_mask = get_pixel_mask(
            exposure.mask,
            self.config.bad_mask_planes,
            dilation=self.config.bad_mask_dilation,
        )
        edges[bad_mask] = False

        return edges, bad_mask

    @timed("detect")
    def detect(self, edges: NDArray[np.bool_]) -> list[Line2D]:
        """Perform linear feature detection on Canny binary edge map.

        Parameters
        ----------
        edges : `numpy.ndarray`, (Ny, Nx)
            The Canny binary edge map.

        Returns
        -------
        lines : `list` [`Line2D`]
            The detected lines, in the centered pixel frame.
        """
        result = lsst.kht.find_lines(
            edges,
            self.config.cluster_minimum_size,
            self.config.cluster_minimum_deviation,
            self.config.delta,
            self.config.minimum_kernel_height,
            self.config.nsigma,
            self.config.abs_minimum_kernel_height,
        )
        self.log.info(f"The Kernel Hough Transform detected {result.size:d} line(s)")

        return [] if result.size == 0 else self._find_clusters(result.rho, result.theta)

    @timed("postprocess")
    def postprocess(
        self,
        streaks: afwTable.SourceTable,
        exposure: afwImage.ExposureF,
        lines: list[Line2D],
    ) -> NDArray[np.bool_]:
        """Perform postprocessing of detected linear features.

        Parameters
        ----------
        streaks : `lsst.afw.table.SourceTable`
            The output streak catalog.
        exposure : `lsst.afw.image.ExposureF`
            The exposure that was searched.
        lines : `list` [`Line2D`]
            The detected lines, in the centered pixel frame.

        Returns
        -------
        streak_mask : `numpy.ndarray`, (Ny, Nx)
            The streak mask plane array.
        """
        box = exposure.getBBox()
        wcs = exposure.getWcs()
        shift = geom.Extent2D(box.getCenter())
        shape = exposure.image.array.shape
        line_masks = [np.zeros(shape, dtype=bool)]

        for kht_line in lines:
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

            # Individual streak mask
            line_mask = get_line_mask(line, shape, self.config.streak_width)
            line_masks.append(line_mask)
        self.log.info(f"Accepted {len(streaks):d} streak(s)")

        streak_mask = np.array(line_masks).any(axis=0)
        if self.config.only_mask_detected:
            streak_mask &= get_pixel_mask(exposure.mask, self.config.detected_mask_plane)

        return streak_mask
        

    def _find_clusters(self, rhos: NDArray[np.float64], thetas: NDArray[np.float64]) -> list[Line2D]:
        """Cluster nearby lines by recursive k-means clustering.

        Parameters
        ----------
        rhos, thetas : `numpy.ndarray`
            The Hesse normal form rho and theta parameters of the detected
            lines.

        Returns
        -------
        lines : `list` [`Line2D`]
            The lines corresponding to the consolidated cluster centers, in
            the centered pixel frame.
        """
        points = np.column_stack((rhos / self.config.rho_bin_size, thetas / self.config.theta_bin_size))
        lines: list[Line2D] = []

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

        for cluster in kmeans.cluster_centers_:
            lines.append(
                Line2D(
                    cluster[0] * self.config.rho_bin_size,
                    cluster[1] * self.config.theta_bin_size * geom.degrees,
                ),
            )
        self.log.info(f"Lines were grouped into {len(lines)} potential streak(s)")

        return lines
