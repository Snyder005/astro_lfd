# First round of testing: Replicate results with following changes
# 1. Change to modern Python instance names (snake_case for variables, CamelCase for classes, etc.)
# 2. Replace the Line functionality with astro_lfd.geom.Line2D functionality.
#    If there are instances where the below code mirrors Line2D methods, check
#    for divergences. For initial test (replication of results) do not replace,
#    but note for improvement for second round of testing.
# 3. Consolidate/refactor/simplify code, but preserve replication of results.
#
# Second round: Using the knowledge from the first task implement code optimizations
# 1. Priority is replace any duplicated code from 2 with Line2D methods, or via
#    additions to the Line2D methods if it fits within its framework as a geometric primitive (can look
#    at Point2D for a list of reasonable methods).
# 2. Additional code optimizations that may not exactly replicate results to pixel and float precision,
#    but allow minimal changes, preserve the fundamental result (line orientation to high precision, pixel
#    level mask need not be exactly one-to-one, but should have minimal changes.
#
# Third round: In the future, modify to operate on SourceRecords in a catalog as in the
# DM Plugin framework. This requires preliminary work to create an appropriate PluginRegistry/StreakPlugin
# base class, etc.
#
__all__ = ["FitProfile", "FitProfileConfig"]

import math

import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
import scipy
from lsst.utils.logging import LsstLogAdapter
from numpy.typing import NDArray

from ..geom.line import Line2D
from .base import get_pixel_mask


class FitProfileConfig(pexConfig.Config):
    """Configurable parameters for `FitProfile`."""

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
        doc="Maximum allowed change (in pixels) of the position parameter is twice this value.",
        dtype=float,
        default=40.0,
    )
    theta_bin_size = pexConfig.Field(
        doc="Maximum allowed change (in degrees) of the angle parameter is twice this value.",
        dtype=float,
        default=2.0,
    )
    inv_sigma = pexConfig.Field(
        doc="Initial inverse of the Moffat sigma parameter (in pixels) describing the streak profile.",
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
        doc="Maximum width (in pixels) of the streak mask; no maximum if 0.",
        dtype=float,
        default=0.0,
    )
    nsigma_mask = pexConfig.Field(
        doc="Factor by which to multiply the fit line sigma to get the final mask width.",
        dtype=float,
        default=5.0,
    )
    footprint_threshold = pexConfig.Field(
        doc="Model value above which pixels are included in the final line mask.",
        dtype=float,
        default=0.01,
    )


class FitProfile:
    """Measure a line by fitting a Moffat profile to the image.

    Refactor of `lsst.meas.algorithms.maskStreaks.LineProfile` and the fitting
    steps of `lsst.meas.algorithms.maskStreaks.MaskStreaksTask`, operating on
    `~astro_lfd.geom.Line2D` inputs.

    Parameters
    ----------
    config : `FitProfileConfig`, optional
        Configuration (default configuration, if not provided).
    log : `lsst.utils.logging.LsstLogAdapter`, optional
        Logger for reporting fit failures.

    Notes
    -----
    Input and output lines are in the PIXEL frame of the exposure (the frame
    of its bounding box).
    """

    ConfigClass = FitProfileConfig

    def __init__(self, config: FitProfileConfig | None = None, log: LsstLogAdapter | None = None):
        self.config = config if config is not None else self.ConfigClass()
        self.log = log

    def measure(self, exposure: afwImage.ExposureF, line: Line2D) -> pipeBase.Struct | None:
        """Fit the line profile.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            The exposure containing the line.
        line : `astro_lfd.geom.Line2D`
            Initial estimate of the line, in the exposure PIXEL frame.

        Returns
        -------
        result : `lsst.pipe.base.Struct` or `None`
            `None` if the fit failed, otherwise a struct with components:

            ``line``
                The fit line, in the exposure PIXEL frame
                (`astro_lfd.geom.Line2D`).
            ``sigma``
                The fit Moffat sigma parameter (`float`).
            ``reduced_chi2``
                The reduced chi2 of the fit (`float`).
            ``model_maximum``
                The peak absolute value of the fit model (`float`).
            ``mask``
                Pixels where the fit model exceeds the footprint threshold
                (`numpy.ndarray`, (Ny, Nx)).
        """
        data = exposure.image.array
        weights = exposure.variance.array**-1
        weights[~np.isfinite(weights) | ~np.isfinite(data)] = 0
        weights[get_pixel_mask(exposure.mask, self.config.bad_mask_planes)] = 0
        detection_mask = get_pixel_mask(exposure.mask, self.config.detected_mask_plane)

        model = _ProfileModel(data, weights, exposure.getBBox())
        sigma = self.config.inv_sigma**-1
        if not model.set_line_mask(line, 20 * sigma, detection_mask=detection_mask, log=self.log):
            return None

        rho, theta = line.rho, line.theta.asRadians()
        x, reduced_chi2, failure = model.fit(
            np.array([rho, theta, sigma**-1]),
            dchi2_tol=self.config.dchi2_tolerance,
            max_iter=self.config.max_fit_iter,
            log=self.log,
        )
        fit_line = Line2D(x[0], x[1] * geom.radians)
        fit_sigma = abs(x[2]) ** -1

        # Initial estimate should be quite close: fit is deemed unsuccessful
        # if the line moves (at the image center) or theta changes by more
        # than twice the allowed bin size. Raw theta is compared, so there is
        # no wrap-around at 0/180 deg.
        shift = model.center_distance(x[0], x[1]) - model.center_distance(rho, theta)
        if (abs(shift) > 2 * self.config.rho_bin_size) or (
            abs(np.rad2deg(x[1] - theta)) > 2 * self.config.theta_bin_size
        ):
            failure = True
            if self.log is not None:
                self.log.debug("Streak fit moved too far from initial estimate.")

        if failure:
            return None

        width = 2 * self.config.nsigma_mask * fit_sigma
        max_width = self.config.max_streak_width
        if (max_width > 0) and (max_width < width):
            if self.log is not None:
                self.log.info("Line with width %d exceeded maximum configured width of %d", width, max_width)
            width = max_width
        model.set_line_mask(fit_line, width)
        final_model = model.make_profile(x[0], x[1], fit_sigma)

        return pipeBase.Struct(
            line=fit_line,
            sigma=fit_sigma,
            reduced_chi2=reduced_chi2,
            model_maximum=abs(final_model).max(),
            mask=abs(final_model) > self.config.footprint_threshold,
        )


class _ProfileModel:
    """Moffat line profile model of the image data near a line.

    Fit parameters are ``rho`` (pixels), ``theta`` (radians), and
    ``1 / sigma`` (inverse pixels) in the PIXEL frame. They are kept raw
    (not canonicalized like `Line2D`), so the derivatives stay continuous
    when ``theta`` crosses 0 or 180 degrees.

    Parameters
    ----------
    data : `numpy.ndarray`, (Ny, Nx)
        The image data.
    weights : `numpy.ndarray`, (Ny, Nx)
        The pixel weights; pixels with zero weight are excluded.
    bbox : `lsst.geom.Box2I`
        PIXEL-frame bounding box of the image.
    """

    BIN_SIZE: float = 5.0
    """Along-line bin size (pixels) for detection and outlier statistics."""

    def __init__(
        self,
        data: NDArray[np.floating],
        weights: NDArray[np.floating],
        bbox: geom.Box2I,
    ):
        self.data = data
        self.weights = weights
        self.bbox = geom.Box2D(bbox)
        self._center = self.bbox.getCenter()
        self._half_diagonal = 0.5 * math.hypot(self.bbox.width, self.bbox.height)
        self._x = np.arange(bbox.getMinX(), bbox.getMaxX() + 1)
        self._y = np.arange(bbox.getMinY(), bbox.getMaxY() + 1)
        self.mask = weights != 0

    def center_distance(self, rho: float, theta: float) -> float:
        """Return the signed distance from the image center to the line."""
        return math.cos(theta) * self._center.x + math.sin(theta) * self._center.y - rho

    def set_line_mask(
        self,
        line: Line2D,
        width: float,
        detection_mask: NDArray[np.bool_] | None = None,
        log: LsstLogAdapter | None = None,
    ) -> bool:
        """Set the mask of pixels near the line used for the fit.

        Parameters
        ----------
        line : `astro_lfd.geom.Line2D`
            The line, in the PIXEL frame.
        width : `float`
            Full width in pixels of the mask.
        detection_mask : `numpy.ndarray`, (Ny, Nx), optional
            Detected pixels; if provided, sections of the line without
            detections are removed from the mask.
        log : `lsst.utils.logging.LsstLogAdapter`, optional
            Logger for reporting.

        Returns
        -------
        success : `bool`
            `False` if the line does not intersect the image or the mask is
            empty.
        """
        distance = line.normal_coordinate_array(self._x[None, :], self._y[:, None])
        self.line_mask = self.mask & (abs(distance) < width / 2)

        if detection_mask is not None:
            if line.clipped_to(self.bbox) is None:
                if log is not None:
                    log.debug("Calculated line not contained in image bounding box")
                return False

            # Mask out areas where there are no detected pixels. This happens
            # when, for example, the streak ends in the middle of the image.
            iy, ix = np.nonzero(self.line_mask)
            if iy.size == 0:
                return False
            s = line.along_coordinate_array(self._x[ix], self._y[iy])
            ndetected, count, bin_index = self._bin_along_line(s, detection_mask[iy, ix])
            detection_fraction = ndetected / np.where(count > 0, count, 1)
            nonzero = detection_fraction[detection_fraction != 0]
            if nonzero.size > 0:
                empty = detection_fraction < 0.5 * np.median(nonzero)
                self.line_mask[iy[empty[bin_index]], ix[empty[bin_index]]] = False

        iy, ix = np.nonzero(self.line_mask)
        self._mask_data = self.data[iy, ix]
        self._mask_weights = self.weights[iy, ix]
        self._mx = self._x[ix]
        self._my = self._y[iy]
        return iy.size > 0

    def _bin_along_line(self, s: NDArray[np.float64], values: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return per-bin sums, counts, and bin indices along the line."""
        bin_index = ((s - s.min()) // self.BIN_SIZE).astype(int)
        sums = np.bincount(bin_index, weights=values)
        counts = np.bincount(bin_index)
        return sums, counts, bin_index

    def _masked_profile(
        self, rho: float, theta: float, inv_sigma: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the masked line model and its derivatives.

        Derivatives are with respect to ``rho``, ``theta``, and
        ``1 / sigma``.
        """
        costheta, sintheta = math.cos(theta), math.sin(theta)
        # Normal and along-line coordinates; the latter is the derivative of
        # the former with respect to theta.
        distance = costheta * self._mx + sintheta * self._my - rho
        along = -sintheta * self._mx + costheta * self._my
        distance_sq = distance**2

        # Moffat profile.
        base = 1 + distance_sq * inv_sigma**2
        profile = base**-2.5
        dprofile = -2.5 * base**-3.5

        weighted_profile = self._mask_weights * profile
        flux = (weighted_profile * self._mask_data).sum() / (weighted_profile * profile).sum()
        if np.isnan(flux):
            flux = 0

        model = flux * profile

        flux_dprofile = flux * dprofile
        flux_dprofile_inv_sigma = flux_dprofile * inv_sigma**2
        dmodel = np.array(
            [
                flux_dprofile_inv_sigma * -2 * distance,
                flux_dprofile_inv_sigma * 2 * distance * along,
                flux_dprofile * distance_sq * 2 * inv_sigma,
            ]
        )
        return model, dmodel

    def make_profile(self, rho: float, theta: float, sigma: float) -> NDArray:
        """Return the line profile model over the full image."""
        model, _ = self._masked_profile(rho, theta, sigma**-1)
        profile = np.zeros(self.data.shape, dtype=self.data.dtype)
        profile[self.line_mask] = model
        return profile

    def _chi2(self, x: NDArray[np.float64], grad: bool = True):
        """Return the reduced chi2 and, optionally, its gradient and Hessian."""
        model, dmodel = self._masked_profile(*x)
        mask_size = (self._mask_weights != 0).sum()
        residual = self._mask_data - model
        chi2 = (self._mask_weights * residual**2).sum()
        if not grad:
            return chi2 / mask_size

        dchi2 = dmodel @ (-2 * self._mask_weights * residual)
        hessian = (dmodel * (2 * self._mask_weights)) @ dmodel.T
        return chi2 / mask_size, dchi2 / mask_size, hessian / mask_size

    def _reject_outliers(self, x: NDArray[np.float64]) -> int:
        """Zero the weight of outlier pixels and return their number.

        This calculates the chi2/dof in bins of pixels perpendicular to the
        streak direction and removes outliers, so that the profile fitter
        ignores regions like the area around bright stars.
        """
        rho, theta, inv_sigma = x
        if Line2D(rho, theta * geom.radians).clipped_to(self.bbox) is None:
            # Line profile is outside the image bounding box.
            return 0

        nonzero = self._mask_weights != 0
        if not nonzero.any():
            return 0

        model, _ = self._masked_profile(rho, theta, inv_sigma)
        pixel_chi2 = (self._mask_weights * (self._mask_data - model) ** 2)[nonzero]
        along = -math.sin(theta) * self._mx[nonzero] + math.cos(theta) * self._my[nonzero]
        chi2, count, bin_index = self._bin_along_line(along, pixel_chi2)

        reduced_chi2 = chi2[count > 0] / count[count > 0]
        valid = reduced_chi2[reduced_chi2 != 0]
        if valid.size == 0:
            return 0
        outliers = np.zeros(chi2.size, dtype=bool)
        outliers[count > 0] = reduced_chi2 > valid.mean() + 3 * valid.std()

        outlier_pixels = outliers[bin_index]
        nonzero_weights = self._mask_weights[nonzero]
        nonzero_weights[outlier_pixels] = 0
        self._mask_weights[nonzero] = nonzero_weights

        return outlier_pixels.sum()

    def fit(
        self,
        x: NDArray[np.float64],
        dchi2_tol: float = 0.1,
        max_iter: int = 100,
        log: LsstLogAdapter | None = None,
    ) -> tuple[NDArray[np.float64], float, bool]:
        """Perform Newton-Raphson minimization to find line parameters.

        Uses the analytic gradient and Hessian with a Brent line search along
        the Newton step at each iteration.

        Parameters
        ----------
        x : `numpy.ndarray`
            Initial parameters ``(rho, theta, 1 / sigma)``.
        dchi2_tol : `float`, optional
            Change in chi2 tolerated for fit convergence.
        max_iter : `int`, optional
            Maximum number of fit iterations allowed.
        log : `lsst.utils.logging.LsstLogAdapter`, optional
            Logger for reporting fit failures.

        Returns
        -------
        x : `numpy.ndarray`
            Fit parameters ``(rho, theta, 1 / sigma)``.
        reduced_chi2 : `float`
            Reduced chi2 at the start of the final iteration.
        failure : `bool`
            `True` if the fit failed.
        """
        x = x.copy()
        dchi2 = 1
        n_iter = 0
        old_chi2 = 0
        n_outliers = 1
        failure = False

        def line_search(c, dx):
            return self._chi2(x - c * dx, grad=False)

        while (abs(dchi2) > dchi2_tol) or (n_outliers != 0):
            params = x.copy()
            chi2, b, A = self._chi2(params)
            if chi2 == 0:
                break
            if not np.isfinite(A).all():
                failure = True
                if log is not None:
                    log.warning("Hessian matrix has non-finite elements.")
                break
            dchi2 = old_chi2 - chi2
            try:
                cholesky = scipy.linalg.cho_factor(A)
            except np.linalg.LinAlgError:
                failure = True
                if log is not None:
                    log.warning("Hessian matrix is not invertible.")
                break
            dx = scipy.linalg.cho_solve(cholesky, b)

            if abs(line_search(1, dx) - chi2) < 1e-12:
                # Step size is too small for the Brent line search to work
                # well; use the step from the Cholesky solve.
                factor = 1
            else:
                factor, *_ = scipy.optimize.brent(line_search, args=(dx,), full_output=True, tol=0.05)

            x -= factor * dx
            if (abs(self.center_distance(x[0], x[1])) > 1.5 * self._half_diagonal) or (n_iter > max_iter):
                failure = True
                break
            old_chi2 = chi2

            n_outliers = self._reject_outliers(params)
            n_iter += 1

        return x, chi2, failure
