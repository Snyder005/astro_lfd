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

        # Peak detector (to be developed fully). Returns Hesse parameters in
        # the binned/padded pixel grid the ADRT actually ran on; the transform
        # is deliberately agnostic to preprocessing.
        rhos, thetas = self._find_peaks(adrt_result)

        # Undo binning here (outside the ADRT coordinate transform): binning is
        # an isotropic rescale of rho only and leaves theta unchanged, so it
        # belongs with the other array-frame -> PIXEL-frame corrections (e.g.
        # XY0) in the caller rather than inside the transform utilities.
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
        rhos, thetas = _adrt_to_hesse(q, h, s, N)

        return rhos, thetas


def _adrt_to_hesse(
    q: NDArray[np.integer] | int,
    h: NDArray[np.floating] | float,
    s: NDArray[np.floating] | float,
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ADRT coordinates to Hesse normal form parameters.

    This is the analytic, vectorized inverse of `_hesse_to_adrt`. It maps the
    ADRT quadrant/height/slope indices ``(q, h, s)`` returned by a peak finder
    directly to Hesse normal ``(rho, theta)`` parameters, reproducing
    `adrt.utils.coord_adrt` in closed form (to floating-point precision) rather
    than building and indexing its full coordinate tables. Being analytic, it
    also accepts *fractional* ``h`` and ``s`` from sub-pixel peak refinement,
    which the integer-indexed table lookup cannot.

    The returned parameters are in the pixel grid the ADRT ran on (the binned,
    zero-padded array). This utility is intentionally agnostic to preprocessing:
    undoing binning and applying any ``XY0`` origin offset to reach the true
    LSST PIXEL frame is the caller's responsibility (see `detect`). Padding is
    already accounted for via ``N`` and only enlarges the domain along the
    bottom rows, so it does not move the origin.

    Parameters
    ----------
    q : `numpy.ndarray` or `int`
        The ADRT result quadrant index/indices (0-3).
    h, s : `numpy.ndarray` or `float`
        The ADRT result height and slope index/indices. May be fractional
        (sub-pixel).
    N : `int`
        Size of the ADRT domain (must be a power of 2).

    Returns
    -------
    rho, theta : `numpy.ndarray`
        The Hesse normal form rho (pixels) and theta (radians) parameters,
        broadcast to the common shape of ``q``, ``h``, ``s``.
    """
    q = np.asarray(q)
    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    c = (N - 1) / 2.0

    # Base digital-line geometry from the slope index (coord_adrt internals).
    ns = s / (N - 1)
    ts = np.arctan(ns)
    cs = np.cos(ts) + np.sin(ts)

    # Fractional Radon-domain offset from the height index.
    hi = 1.0 - (2.0 * h + 1.0) / (2.0 * N)
    h0 = ((hi + ((2.0 * N - 1.0) / (2.0 * N)) * ns) / (1.0 + ns) - 0.5) * cs

    # Quadrant selects the sign of the offset and the line angle. Even
    # quadrants (0, 2) keep the offset sign; odd quadrants (1, 3) flip it.
    offset = np.where(q % 2 == 0, h0, -h0)
    angle = np.select(
        [q == 0, q == 1, q == 2, q == 3],
        [ts - np.pi / 2.0, -ts, ts, np.pi / 2.0 - ts],
    )

    # Line angle -> normal-vector angle, then Radon offset -> PIXEL rho with the
    # image-center -> corner-origin recenter (see devel/adrt_coordinate_transform.md).
    theta = np.pi / 2.0 - angle
    rho = -offset * N + c * (np.cos(theta) + np.sin(theta))

    return rho, theta


def _hesse_to_adrt(
    rho: NDArray[np.floating] | float,
    theta: NDArray[np.floating] | float,
    N: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Hesse normal form parameters to ADRT coordinates.

    Analytic, vectorized inverse of `_adrt_to_hesse`: maps Hesse normal
    ``(rho, theta)`` (in the ADRT's binned/padded pixel grid) back to ADRT
    quadrant/height/slope indices ``(q, h, s)``. Returns floating-point indices
    so sub-pixel positions are preserved.

    ``theta`` is reduced modulo ``pi`` (Hesse lines are undirected). The
    round-trip is exact in the interior; it is degenerate only on the
    quadrant-boundary angles (0, 45, 90, 135 deg, i.e. ``s == 0`` and
    ``s == N - 1``), where adjacent ADRT quadrants share the same slope and the
    quadrant assignment is a convention choice.

    Parameters
    ----------
    rho, theta : `numpy.ndarray` or `float`
        The Hesse normal form rho (pixels) and theta (radians) parameters.
    N : `int`
        Size of the ADRT domain (must be a power of 2).

    Returns
    -------
    q, h, s : `numpy.ndarray`
        The ADRT quadrant, height, and slope indices, broadcast to the common
        shape of ``rho`` and ``theta``. ``q`` is integer-valued; ``h`` and
        ``s`` may be fractional.
    """
    rho = np.asarray(rho, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    c = (N - 1) / 2.0

    # Reduce to [0, pi) and pick the quadrant + base line angle. The four ADRT
    # quadrants tile [0, pi) in normal-vector angle as: [0, pi/4)->3,
    # [pi/4, pi/2)->2, [pi/2, 3pi/4)->1, [3pi/4, pi)->0.
    th = np.mod(theta, np.pi)
    conds = [th < np.pi / 4.0, th < np.pi / 2.0, th < 3.0 * np.pi / 4.0]
    q = np.select(conds, [3, 2, 1], default=0)
    ts = np.select(
        conds,
        [th, np.pi / 2.0 - th, th - np.pi / 2.0],
        default=np.pi - th,
    )

    # Invert the slope geometry.
    ns = np.tan(ts)
    s = ns * (N - 1)
    cs = np.cos(ts) + np.sin(ts)

    # Invert the rho recenter/scale to recover the Radon offset, undo the
    # quadrant sign flip, then invert the height mapping. Trig uses `th` so it
    # is consistent with the quadrant reduction above.
    offset = (c * (np.cos(th) + np.sin(th)) - rho) / N
    h0 = np.where(q % 2 == 0, offset, -offset)
    hi = (h0 / cs + 0.5) * (1.0 + ns) - ((2.0 * N - 1.0) / (2.0 * N)) * ns
    h = N * (1.0 - hi) - 0.5

    return q.astype(np.float64), h, s
