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
    Line parameters are in the frame of the original ``LineProfile`` code: the
    origin is at the center of the image array, ``(nx / 2, ny / 2)``.
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
            Initial estimate of the line, with origin at the image center.

        Returns
        -------
        result : `lsst.pipe.base.Struct` or `None`
            `None` if the fit failed, otherwise a struct with components:

            ``line``
                The fit line (`astro_lfd.geom.Line2D`).
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

        # Line2D is canonicalized, so the raw fit parameters are used in the
        # step-size check below.
        rho = float(line.rho)
        theta = float(line.theta.asDegrees())
        sigma = self.config.inv_sigma**-1

        model = _ProfileModel(data, weights)
        if not model.set_line_mask(rho, theta, sigma, 0, 10, detection_mask=detection_mask, log=self.log):
            return None
        if model.line_mask.sum() == 0:
            return None

        x, reduced_chi2, failure = model.fit(
            np.array([rho, theta, sigma**-1]),
            dchi2_tol=self.config.dchi2_tolerance,
            max_iter=self.config.max_fit_iter,
            log=self.log,
        )
        fit_rho, fit_theta, fit_sigma = x[0], x[1], abs(x[2]) ** -1

        # Initial estimate should be quite close: fit is deemed unsuccessful
        # if rho or theta change by more than twice the allowed bin size.
        if (abs(fit_rho - rho) > 2 * self.config.rho_bin_size) or (
            abs(fit_theta - theta) > 2 * self.config.theta_bin_size
        ):
            failure = True
            if self.log is not None:
                self.log.debug("Streak fit moved too far from initial estimate.")

        if failure:
            return None

        model.set_line_mask(
            fit_rho, fit_theta, fit_sigma, self.config.max_streak_width, self.config.nsigma_mask, log=self.log
        )
        final_model = model.make_profile(fit_rho, fit_theta, fit_sigma)

        return pipeBase.Struct(
            line=Line2D(fit_rho, fit_theta * geom.degrees),
            sigma=fit_sigma,
            reduced_chi2=reduced_chi2,
            model_maximum=abs(final_model).max(),
            mask=abs(final_model) > self.config.footprint_threshold,
        )


class _ProfileModel:
    """Moffat line profile model of the image data near a line.

    Line parameters are ``rho`` (pixels, from the image center), ``theta``
    (degrees), and ``sigma`` (pixels).

    Parameters
    ----------
    data : `numpy.ndarray`, (Ny, Nx)
        The image data.
    weights : `numpy.ndarray`, (Ny, Nx)
        The pixel weights; pixels with zero weight are excluded.
    """

    def __init__(self, data: NDArray[np.floating], weights: NDArray[np.floating]):
        self.data = data
        self.weights = weights
        self._ymax, self._xmax = data.shape
        self._rho_max = ((0.5 * self._ymax) ** 2 + (0.5 * self._xmax) ** 2) ** 0.5
        xrange = np.arange(self._xmax) - self._xmax / 2.0
        yrange = np.arange(self._ymax) - self._ymax / 2.0
        self._xmesh, self._ymesh = np.meshgrid(xrange, yrange)
        self.mask = weights != 0

    def _line_ends(self, rho: float, theta: float) -> NDArray[np.float64]:
        """Return the pixel coordinates of the line ends at the image edges.

        Coordinates have their origin at the array corner (not center).
        """
        # TODO (Step 2): equivalent to Line2D.boundary_intersections, but the
        # point ordering (and therefore which end is first) differs.
        radtheta = np.deg2rad(theta)
        costheta = np.cos(radtheta)
        sintheta = np.sin(radtheta)
        # Bottom, left, top, and right box edge intersections.
        ya = -self._ymax / 2.0
        xb = -self._xmax / 2.0
        yc = self._ymax / 2.0
        xd = self._xmax / 2.0
        intersections = np.array(
            [
                [(rho - ya * sintheta) / costheta, ya],
                [xb, (rho - xb * costheta) / sintheta],
                [(rho - yc * sintheta) / costheta, yc],
                [xd, (rho - xd * costheta) / sintheta],
            ]
        )
        intersections[:, 0] += self._xmax / 2.0
        intersections[:, 1] += self._ymax / 2.0

        in_box = (
            (intersections[:, 0] >= 0)
            & (intersections[:, 0] <= self._xmax)
            & (intersections[:, 1] >= 0)
            & (intersections[:, 1] <= self._ymax)
        )
        return intersections[in_box]

    def _distance_from_line_end(
        self, theta: float, line_end: NDArray[np.float64], xmesh: NDArray, ymesh: NDArray
    ) -> NDArray[np.float64]:
        """Return the along-line distance of pixels from a line end."""
        # TODO (Step 2): equivalent to Line2D.along_coordinate up to an offset.
        xa = line_end[0] - self._xmax / 2.0
        ya = line_end[1] - self._ymax / 2.0
        radtheta = np.deg2rad(theta)
        costheta = np.cos(radtheta)
        sintheta = np.sin(radtheta)
        return -sintheta * xmesh + costheta * ymesh + sintheta * xa - costheta * ya

    def set_line_mask(
        self,
        rho: float,
        theta: float,
        sigma: float,
        max_streak_width: float,
        nsigma_mask: float,
        detection_mask: NDArray[np.bool_] | None = None,
        log: LsstLogAdapter | None = None,
    ) -> bool:
        """Set the mask of pixels near the line used for the fit.

        Parameters
        ----------
        rho, theta, sigma : `float`
            The line profile parameters.
        max_streak_width : `float`
            Maximum width in pixels of the mask; no maximum if 0.
        nsigma_mask : `float`
            Factor by which to multiply ``sigma`` to get the mask width.
        detection_mask : `numpy.ndarray`, (Ny, Nx), optional
            Detected pixels; if provided, sections of the line without
            detections are removed from the mask.
        log : `lsst.utils.logging.LsstLogAdapter`, optional
            Logger for reporting.

        Returns
        -------
        success : `bool`
            `False` if the line does not intersect the image.
        """
        # TODO (Step 2): distance is Line2D.normal_coordinate on the mesh.
        radtheta = np.deg2rad(theta)
        distance = np.cos(radtheta) * self._xmesh + np.sin(radtheta) * self._ymesh - rho

        width = 2 * nsigma_mask * sigma
        if (max_streak_width > 0) and (max_streak_width < width):
            if log is not None:
                log.info(
                    "Line with width %d exceeded maximum configured width of %d", width, max_streak_width
                )
            width = max_streak_width
        self.line_mask = self.mask & (abs(distance) < width / 2)

        if detection_mask is not None:
            # Mask out areas where there are no detected pixels. This happens
            # when, for example, the streak ends in the middle of the image.
            line_ends = self._line_ends(rho, theta)
            if line_ends.size == 0:
                if log is not None:
                    log.debug("Calculated line not contained in image bounding box")
                return False

            detections = detection_mask[self.line_mask] != 0
            distance_from_end = self._distance_from_line_end(
                theta, line_ends[0], self._xmesh[self.line_mask], self._ymesh[self.line_mask]
            )
            bins = np.arange(distance_from_end.min(), distance_from_end.max() + 5.1, 5)
            ndetected, _, bin_number = scipy.stats.binned_statistic(
                distance_from_end, detections, statistic="sum", bins=bins
            )
            count, *_ = scipy.stats.binned_statistic(distance_from_end, detections, statistic="count", bins=bins)
            detection_fraction = ndetected / count
            empty = detection_fraction < (0.5 * np.median(detection_fraction[detection_fraction != 0]))
            self.line_mask[self.line_mask] = ~empty[bin_number - 1]

        self._mask_data = self.data[self.line_mask]
        self._mask_weights = self.weights[self.line_mask]
        self._mxmesh = self._xmesh[self.line_mask]
        self._mymesh = self._ymesh[self.line_mask]
        return True

    def _masked_profile(
        self, rho: float, theta: float, sigma: float, fit_flux: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the masked line model and its derivatives.

        Derivatives are with respect to ``rho``, ``theta``, and
        ``1 / sigma``.
        """
        inv_sigma = sigma**-1
        radtheta = np.deg2rad(theta)
        costheta = np.cos(radtheta)
        sintheta = np.sin(radtheta)
        distance = costheta * self._mxmesh + sintheta * self._mymesh - rho
        distance_sq = distance**2

        # Partial derivatives of the squared distance.
        drad = np.pi / 180
        ddist_sq_drho = 2 * distance * (-np.ones_like(self._mxmesh))
        ddist_sq_dtheta = 2 * distance * (-sintheta * self._mxmesh + costheta * self._mymesh) * drad

        # Moffat profile.
        profile = (1 + distance_sq * inv_sigma**2) ** -2.5
        dprofile = -2.5 * (1 + distance_sq * inv_sigma**2) ** -3.5

        if fit_flux:
            flux = (self._mask_weights * self._mask_data * profile).sum() / (
                self._mask_weights * profile**2
            ).sum()
        else:
            # Approximately normalize the line.
            flux = inv_sigma**-1
        if np.isnan(flux):
            flux = 0

        model = flux * profile

        flux_dprofile = flux * dprofile
        flux_dprofile_inv_sigma = flux_dprofile * inv_sigma**2
        dmodel = np.array(
            [
                flux_dprofile_inv_sigma * ddist_sq_drho,
                flux_dprofile_inv_sigma * ddist_sq_dtheta,
                flux_dprofile * distance_sq * 2 * inv_sigma,
            ]
        )
        return model, dmodel

    def make_profile(self, rho: float, theta: float, sigma: float, fit_flux: bool = True) -> NDArray:
        """Return the line profile model over the full image."""
        model, _ = self._masked_profile(rho, theta, sigma, fit_flux=fit_flux)
        profile = np.zeros((self._ymax, self._xmax), dtype=self.data.dtype)
        profile[self.line_mask] = model
        return profile

    def _chi2(self, rho: float, theta: float, sigma: float, grad: bool = True):
        """Return the reduced chi2 and, optionally, its gradient and Hessian.

        Derivatives are with respect to ``rho``, ``theta``, and
        ``1 / sigma``.
        """
        model, dmodel = self._masked_profile(rho, theta, sigma)
        mask_size = (self._mask_weights != 0).sum()
        chi2 = (self._mask_weights * (self._mask_data - model) ** 2).sum()
        if not grad:
            return chi2 / mask_size

        dchi2 = ((-2 * self._mask_weights * (self._mask_data - model))[None, :] * dmodel).sum(axis=1)
        hessian = (2 * self._mask_weights * dmodel[:, None, :] * dmodel[None, :, :]).sum(axis=2)
        return chi2 / mask_size, dchi2 / mask_size, hessian / mask_size

    def _reject_outliers(self, rho: float, theta: float, sigma: float) -> int:
        """Zero the weight of outlier pixels and return their number.

        This calculates the chi2/dof in bins of pixels perpendicular to the
        streak direction and removes outliers, so that the profile fitter
        ignores regions like the area around bright stars.
        """
        model, _ = self._masked_profile(rho, theta, sigma)
        pixel_chi2 = self._mask_weights * (self._mask_data - model) ** 2

        line_ends = self._line_ends(rho, theta)
        if line_ends.shape != (2, 2):
            # Line profile is outside the image bounding box.
            return 0

        nonzero = self._mask_weights != 0
        distance_from_end = self._distance_from_line_end(theta, line_ends[0], self._mxmesh, self._mymesh)
        distance_from_end = distance_from_end[nonzero]
        pixel_chi2 = pixel_chi2[nonzero]
        bins = np.arange(distance_from_end.min(), distance_from_end.max() + 5.1, 5)
        chi2, _, bin_number = scipy.stats.binned_statistic(
            distance_from_end, pixel_chi2, statistic="sum", bins=bins
        )
        count, *_ = scipy.stats.binned_statistic(distance_from_end, pixel_chi2, statistic="count", bins=bins)

        reduced_chi2 = chi2 / count
        valid = reduced_chi2[reduced_chi2 != 0]
        outliers = reduced_chi2 > (np.nanmean(valid) + 3 * np.nanstd(valid))

        outlier_pixels = outliers[bin_number - 1]
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
            test_x = x - c * dx
            return self._chi2(test_x[0], test_x[1], test_x[2] ** -1, grad=False)

        while (abs(dchi2) > dchi2_tol) or (n_outliers != 0):
            params = (x[0], x[1], x[2] ** -1)
            chi2, b, A = self._chi2(*params)
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
            if (abs(x[0]) > 1.5 * self._rho_max) or (n_iter > max_iter):
                failure = True
                break
            old_chi2 = chi2

            n_outliers = self._reject_outliers(*params)
            n_iter += 1

        return x, chi2, failure
