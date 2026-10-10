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

import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.meas.algorithms.maskStreaks import Line


class FitProfilePlugin:

    def __init__(
        self,
        line: Line,
        exposure: afwImage.ExposureF,
        bad_mask_planes: str | list[str],
        detected_mask_plane: str = "DETECTED",
    ) -> pipeBase.Struct:
        self.data = exposure.image.array
        bad_mask = get_pixel_mask(exposure.mask, bad_mask_planes)
        detected_mask = get_pixel_mask(exposure.mask, detected_mask_plane)
        self.weights = exposure.variance.array**-1
        self.weights[~np.isfinite(self.weights) | ~np.isfinite(data)] = 0
        self.weights[bad_mask] = 0

        # Any of these needed or just convenience to pass between methods?
        self._ymax, self._xmax = data.shape
        self._dtype = data.dtype
        xrange = np.arange(_xmax) - _xmax / 2.
        yrange = np.arange(_ymax) - _ymax / 2.
        self._rhoMax = ((0.5 * self._ymax)**2 + (0.5 * self._xmax)**2)**0.5
        self._xmesh, self._ymesh = np.meshgrid(xrange, yrange)
        self.mask = (weights != 0)

        self._initLine = line
        self.modelFailure = False
        self.setLineMask(line, maxStreakWidth=0, nSigmaMask=10, detectionMask=detected_mask)
        
    def getLineXY(self, line):
        """Return the pixel coordinates of the ends of the line.

        Parameters
        ----------
        line : `Line`
            Line for which to find the endpoints.

        Returns
        -------
        boxIntersections : `np.ndarray`
            (x, y) coordinates of the start and endpoints of the line.
        """
        theta = line.theta * u.deg
        # Determine where the line intersects with each edge of the bounding
        # box.
        # Bottom:
        yA = -self._ymax / 2.
        xA = (line.rho - yA * np.sin(theta)) / np.cos(theta)
        # Left:
        xB = -self._xmax / 2.
        yB = (line.rho - xB * np.cos(theta)) / np.sin(theta)
        # Top:
        yC = self._ymax / 2.
        xC = (line.rho - yC * np.sin(theta)) / np.cos(theta)
        # Right:
        xD = self._xmax / 2.
        yD = (line.rho - xD * np.cos(theta)) / np.sin(theta)

        lineIntersections = np.array([[xA, yA],
                                      [xB, yB],
                                      [xC, yC],
                                      [xD, yD]])
        lineIntersections[:, 0] += self._xmax / 2.
        lineIntersections[:, 1] += self._ymax / 2.

        # The line will necessarily intersect with exactly two edges of the
        # bounding box itself.
        inBox = ((lineIntersections[:, 0] >= 0) & (lineIntersections[:, 0] <= self._xmax)
                 & (lineIntersections[:, 1] >= 0) & (lineIntersections[:, 1] <= self._ymax))
        boxIntersections = lineIntersections[inBox]

        return boxIntersections

    def setLineMask(self, line, maxStreakWidth, nSigmaMask, logger=None, detectionMask=None):
        """Set mask around the image region near the line.

        Parameters
        ----------
        line : `Line`
            Parameters of line in the image.
        maxStreakWidth : `float`
            Maximum width in pixels of streak mask.
        nSigmaMask : `float`
            Factor by which to multiply the line's width to get the mask width.
        logger : `lsst.utils.logging.LsstLogAdapter`, optional
            Logger to use for reporting when maxStreakWidth is reached.
        detectionMask : `np.ndarray`, optional
            2-d boolean array where detected pixels are True.
        """
        if line:
            # Only fit pixels within nSigmaMask * sigma of the estimated line
            radtheta = np.deg2rad(line.theta)
            distance = (np.cos(radtheta) * self._xmesh + np.sin(radtheta) * self._ymesh - line.rho)

            width = 2 * nSigmaMask * line.sigma
            if (maxStreakWidth > 0) and (maxStreakWidth < width):
                if logger is not None:
                    logger.info("Line with width %d exceeded maximum configured width of %d",
                                width, maxStreakWidth)
                width = maxStreakWidth
            m = (abs(distance) < width/2)
            self.lineMask = self.mask & m
            if detectionMask is not None:
                # Mask out areas where there are no detected pixels. This
                # happens when, for example, the streak ends in the middle of
                # the image.
                lineEnds = self.getLineXY(line)
                if lineEnds.size == 0:
                    if logger is not None:
                        logger.debug("Calculated line not contained in image bounding box")
                    self.modelFailure = True
                    return
                xA = lineEnds[0, 0] - self._xmax / 2.
                yA = lineEnds[0, 1] - self._ymax / 2.

                radtheta = np.deg2rad(line.theta)
                costheta = np.cos(radtheta)
                sintheta = np.sin(radtheta)

                maskDetections = detectionMask[self.lineMask] != 0
                distanceFromLineEnd = (- sintheta * self._xmesh[self.lineMask]
                                       + costheta * self._ymesh[self.lineMask]
                                       + sintheta * xA
                                       - costheta * yA)
                lineBins = np.arange(distanceFromLineEnd.min(), distanceFromLineEnd.max() + 5.1, 5)
                # Get the chi2 of the pixels perpendicular to the streak:
                detectionsAlongStreak, _, binnumber = scipy.stats.binned_statistic(distanceFromLineEnd,
                                                                                   maskDetections,
                                                                                   statistic='sum',
                                                                                   bins=lineBins)
                countAlongStreak, *_ = scipy.stats.binned_statistic(distanceFromLineEnd, maskDetections,
                                                                    statistic='count', bins=lineBins)
                detectionFraction = detectionsAlongStreak / countAlongStreak
                emptyRows = detectionFraction < (0.5 * np.median(detectionFraction[detectionFraction != 0]))
                emptyDetections = emptyRows[binnumber - 1]
                self.lineMask[self.lineMask] = ~emptyDetections
        else:
            self.lineMask = np.copy(self.mask)

        self._maskData = self.data[self.lineMask]
        self._maskWeights = self.weights[self.lineMask]
        self._mxmesh = self._xmesh[self.lineMask]
        self._mymesh = self._ymesh[self.lineMask]

    def _makeMaskedProfile(self, line, fitFlux=True):
        """Construct the line model in the masked region and calculate its
        derivatives.

        Parameters
        ----------
        line : `Line`
            Parameters of line profile for which to make profile in the masked
            region.
        fitFlux : `bool`
            Fit the amplitude of the line profile to the data.

        Returns
        -------
        model : `np.ndarray`
            Model in the masked region.
        dModel : `np.ndarray`
            Derivative of the model in the masked region.
        """
        invSigma = line.sigma**-1
        # Calculate distance between pixels and line
        radtheta = np.deg2rad(line.theta)
        costheta = np.cos(radtheta)
        sintheta = np.sin(radtheta)
        distance = (costheta * self._mxmesh + sintheta * self._mymesh - line.rho)
        distanceSquared = distance**2

        # Calculate partial derivatives of distance
        drad = np.pi / 180
        dDistanceSqdRho = 2 * distance * (-np.ones_like(self._mxmesh))
        dDistanceSqdTheta = (2 * distance * (-sintheta * self._mxmesh + costheta * self._mymesh) * drad)

        # Use pixel-line distances to make Moffat profile
        profile = (1 + distanceSquared * invSigma**2)**-2.5
        dProfile = -2.5 * (1 + distanceSquared * invSigma**2)**-3.5

        if fitFlux:
            # Calculate line flux from profile and data
            flux = ((self._maskWeights * self._maskData * profile).sum()
                    / (self._maskWeights * profile**2).sum())
        else:
            # Approximately normalize the line
            flux = invSigma**-1
        if np.isnan(flux):
            flux = 0

        model = flux * profile

        # Calculate model derivatives
        fluxdProfile = flux * dProfile
        fluxdProfileInvSigma = fluxdProfile * invSigma**2
        dModeldRho = fluxdProfileInvSigma * dDistanceSqdRho
        dModeldTheta = fluxdProfileInvSigma * dDistanceSqdTheta
        dModeldInvSigma = fluxdProfile * distanceSquared * 2 * invSigma

        dModel = np.array([dModeldRho, dModeldTheta, dModeldInvSigma])
        return model, dModel

    def makeProfile(self, line, fitFlux=True):
        """Construct the line profile model.

        Parameters
        ----------
        line : `Line`
            Parameters of the line profile to model.
        fitFlux : `bool`, optional
            Fit the amplitude of the line profile to the data.

        Returns
        -------
        finalModel : `np.ndarray`
            Model for line profile.
        """
        model, _ = self._makeMaskedProfile(line, fitFlux=fitFlux)
        finalModel = np.zeros((self._ymax, self._xmax), dtype=self._dtype)
        finalModel[self.lineMask] = model
        return finalModel

    def _lineChi2(self, line, grad=True):
        """Construct the chi2 between the data and the model.

        Parameters
        ----------
        line : `Line`
            `Line` parameters for which to build model and calculate chi2.
        grad : `bool`, optional
            Whether or not to return the gradient and hessian.

        Returns
        -------
        reducedChi : `float`
            Reduced chi2 of the model.
        reducedDChi : `np.ndarray`
            Derivative of the chi2 with respect to rho, theta, invSigma.
        reducedHessianChi : `np.ndarray`
            Hessian of the chi2 with respect to rho, theta, invSigma.
        """
        # Calculate chi2
        model, dModel = self._makeMaskedProfile(line)
        chi2 = (self._maskWeights * (self._maskData - model)**2).sum()
        maskSize = (self._maskWeights != 0).sum()
        if not grad:
            return chi2.sum() / maskSize

        # Calculate derivative and Hessian of chi2
        derivChi2 = ((-2 * self._maskWeights * (self._maskData - model))[None, :] * dModel).sum(axis=1)
        hessianChi2 = (2 * self._maskWeights * dModel[:, None, :] * dModel[None, :, :]).sum(axis=2)

        reducedChi = chi2 / maskSize
        reducedDChi = derivChi2 / maskSize
        reducedHessianChi = hessianChi2 / maskSize
        return reducedChi, reducedDChi, reducedHessianChi

    def _rejectOutliers(self, line):
        """Reject outlier pixels.

        This calculates the chi2/dof in bins of pixels perpendicular to the
        streak direction and removes outliers. This is done so that the profile
        fitter ignores regions like the area around bright stars.

        Parameters
        ----------
        line : `Line`
            `Line` parameters for which to build model and calculate chi2.

        Returns
        -------
        nOutliers : `int`
            Number of outlier pixels.
        """
        model, _ = self._makeMaskedProfile(line)
        pixelChi2 = (self._maskWeights * (self._maskData - model)**2)

        lineEnds = self.getLineXY(line)

        if lineEnds.shape == (2, 2):
            xA = lineEnds[0, 0] - self._xmax / 2.
            yA = lineEnds[0, 1] - self._ymax / 2.
        else:
            # Line profile is outside the detector bounding box. Exit outlier rejection.
            return 0

        radtheta = np.deg2rad(line.theta)
        costheta = np.cos(radtheta)
        sintheta = np.sin(radtheta)

        distanceFromLineEnd = (- sintheta * self._mxmesh + costheta * self._mymesh + sintheta * xA
                               - costheta * yA)

        distanceFromLineEnd = distanceFromLineEnd[self._maskWeights != 0]
        nonZeroPixelChi2 = pixelChi2[self._maskWeights != 0]
        lineBins = np.arange(distanceFromLineEnd.min(), distanceFromLineEnd.max() + 5.1, 5)
        # Get the chi2 of the pixels perpendicular to the streak:
        chi2AlongStreak, _, binnumber = scipy.stats.binned_statistic(distanceFromLineEnd, nonZeroPixelChi2,
                                                                     statistic='sum', bins=lineBins)
        countAlongStreak, *_ = scipy.stats.binned_statistic(distanceFromLineEnd, nonZeroPixelChi2,
                                                            statistic='count', bins=lineBins)

        rChi2AlongStreak = chi2AlongStreak / countAlongStreak
        outliers = rChi2AlongStreak > (np.nanmean(rChi2AlongStreak[rChi2AlongStreak != 0])
                                       + 3 * np.nanstd(rChi2AlongStreak[rChi2AlongStreak != 0]))

        outlierPix = outliers[binnumber - 1]
        tmpWeights = self._maskWeights[self._maskWeights != 0]
        tmpWeights[outlierPix] = 0
        self._maskWeights[self._maskWeights != 0] = tmpWeights

        nOutliers = outlierPix.sum()

        return nOutliers

    def fit(self, dChi2Tol=0.1, maxIter=100, log=None):
        """Perform Newton-Raphson minimization to find line parameters.

        This method takes advantage of having known derivative and Hessian of
        the multivariate function to quickly and efficiently find the minimum.
        This is more efficient than the scipy implementation of the Newton-
        Raphson method, which doesn't take advantage of the Hessian matrix. The
        method here also performs a line search in the direction of the steepest
        derivative at each iteration, which reduces the number of iterations
        needed.

        Parameters
        ----------
        dChi2Tol : `float`, optional
            Change in Chi2 tolerated for fit convergence.
        maxIter : `int`, optional
            Maximum number of fit iterations allowed. The fit should converge in
            ~10 iterations, depending on the value of dChi2Tol, but this
            maximum provides a backup.
        log : `lsst.utils.logging.LsstLogAdapter`, optional
            Logger to use for reporting more details for fitting failures.

        Returns
        -------
        outline : `Line`
            Coordinates, inverse width, and chi2 of fit line.
        fitFailure : `bool`
            Boolean where `False` corresponds to a successful  fit.
        """
        # Do minimization on inverse of sigma to simplify derivatives:
        x = np.array([self._initLine.rho, self._initLine.theta, self._initLine.sigma**-1])

        dChi2 = 1
        iter = 0
        oldChi2 = 0
        nOutliers = 1
        fitFailure = False

        def line_search(c, dx):
            testx = x - c * dx
            testLine = Line(testx[0], testx[1], testx[2]**-1)
            return self._lineChi2(testLine, grad=False)

        while (abs(dChi2) > dChi2Tol) or (nOutliers != 0):
            line = Line(x[0], x[1], x[2]**-1)
            chi2, b, A = self._lineChi2(line)
            if chi2 == 0:
                break
            if not np.isfinite(A).all():
                fitFailure = True
                if log is not None:
                    log.warning("Hessian matrix has non-finite elements.")
                break
            dChi2 = oldChi2 - chi2
            try:
                cholesky = scipy.linalg.cho_factor(A)
            except np.linalg.LinAlgError:
                fitFailure = True
                if log is not None:
                    log.warning("Hessian matrix is not invertible.")
                break
            dx = scipy.linalg.cho_solve(cholesky, b)

            if abs(line_search(1, dx) - chi2) < 1e-12:
                # Step size is too small for the brent line search to work well.
                # Just use the step fit from the Cholesky solve.
                factor = 1
            else:
                factor, _, _, _ = scipy.optimize.brent(line_search, args=(dx,), full_output=True, tol=0.05)

            x -= factor * dx
            if (abs(x[0]) > 1.5 * self._rhoMax) or (iter > maxIter):
                fitFailure = True
                break
            oldChi2 = chi2

            nOutliers = self._rejectOutliers(line)
            iter += 1

        outline = Line(x[0], x[1], abs(x[2])**-1, chi2)
        return outline, fitFailure

if __name__ == "__main__":

    inv_sigma = 10.**-1
    bad_mask_planes = ["NO_DATA", "INTRP", "BAD", "SAT", "EDGE"]
    detected_mask_plane = "DETECTED"
    rhoBinSize = 
    thetaBinSize = 

    line = Line(500, 1000)
    line.sigma = inv_sigma**-1
    lineModel = FitProfilePlugin(line, exposure, bad_mask_planes, detected_mask_plane)

    if lineModel.modelFailure or lineModel.lineMask.sum() == 0:
        print("Failure")
        return

        fit, fitFailure = lineModel.fit(dChi2Tol=self.config.dChi2Tolerance, log=self.log,
                                        maxIter=self.config.maxFitIter)

        # Initial estimate should be quite close: fit is deemed unsuccessful if rho or theta
        # change more than the allowed bin in rho or theta:
        if ((abs(fit.rho - line.rho) > 2 * rhoBinSize)
                or (abs(fit.theta - line.theta) > 2 * thetaBinSize)):
            fitFailure = True
            self.log.debug("Streak fit moved too far from initial estimate. Line will be dropped.")

        if fitFailure:
            nFitFailures += 1
            continue

        # Make mask
        lineModel.setLineMask(fit, self.config.maxStreakWidth, self.config.nSigmaMask, logger=self.log)
        finalModel = lineModel.makeProfile(fit)
