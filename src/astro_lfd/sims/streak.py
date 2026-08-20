import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass
class Streak:
    """Streak geometry and brightness in Hesse normal form.

    Parameters
    ----------
    rho : `float`
        The signed perpendicular distance from the origin to the line, in
        pixels.
    theta : `float`
        The angle of the line normal vector, in degrees.
    peak_signal : `float`
        The signal at the line center.
    width : `float`
        The top-hat full width of the streak, in pixels.
    """

    rho: float
    theta: float
    peak_signal: float
    width: float

    def get_signal(
        self,
        shape: tuple[int, int],
        fwhm: float | None = None
    ) -> NDArray[np.float64]:
        """Calculate the noise-free streak signal at each pixel from its
        distance from the line.

        Parameters
        ----------
        shape : `tuple` [`int`]
            A tuple of the array dimensions.
        fwhm : `float` or None
            The PSF full-width-at-half-maximum, in pixels (None, by default).

        Returns
        -------
        signal : `numpy.ndarray`
            Noise-free signal at each pixel.
        """
        ny, nx = shape
        gy, gx = np.ogrid[:ny, :nx]
        theta = np.deg2rad(self.theta)
        distance = gx * np.cos(theta) + gy * np.sin(theta) - self.rho

        if fwhm is None:
            return self._box(distance) * self.peak_signal

        else:
            sigma = fwhm * FWHM_TO_SIGMA
            return self._blurred_box(distance, sigma) * self.peak_signal

    def _box(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Top-hat cross-sectional profile, peak-normalized to 1.

        Parameters
        ----------
        d : `numpy.ndarray`
            The signed perpendicular distance from the line, in pixels.

        Returns
        -------
        normalized_profile : `numpy.ndarray`
            Normalized profile values at each distance. Equal to 1 along the
            top-hat width.
        """
        return (np.abs(d) <= self.width / 2).astype(np.float64)

    def _blurred_box(
        self,
        d: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64]:
        """Top-hat convolved with a Gaussian, peak-normalized to 1.

        Parameters
        ----------
        d : `numpy.ndarray`
            The signed perpendicular distance from the line, in pixels.
        sigma : `float`
            Gaussian sigma, in pixels.

        Returns
        -------
        normalized_profile : `numpy.ndarray`
            Normalized profile values at each distance. Equal to 1 along the
            line.
        """
        w = self.width
        y = 0.5 * (erf((d + w / 2) / (np.sqrt(2) * sigma)) - erf((d - w / 2) / (np.sqrt(2) * sigma)))
        return y / erf(w / (2 * np.sqrt(2) * sigma))
