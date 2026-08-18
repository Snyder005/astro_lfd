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
    theta : `float`
        The angle of the line normal vector, in degrees.
    rho : `float`
        The signed perpendicular distance from the origin to the line, in
        pixels.
    width : `float`
        Top-hat full width of the streak, in pixels.
    peak_signal : `float`
        Signal at the ridge center.
    """

    rho: float
    theta: float
    peak_signal: float
    width: float

    def get_signal(
        self,
        fwhm: float,
        shape: tuple[int, int],
    ) -> NDArray[np.float64]:
        """Calculate the noise-free streak signal at each pixel from its
        distance from the line.

        Parameters
        ----------
        fwhm : `float`
            The PSF full-width-at-half-maximum, in pixels.
        shape : `tuple` [`int`]
            A tuple of the array dimensions.

        Returns
        -------
        signal : `numpy.ndarray`
            Noise-free signal at each pixel.
        """
        ny, nx = shape
        gy, gx = np.ogrid[:ny, :nx]
        theta = np.deg2rad(self.theta)
        distance = gx * np.cos(theta) + gy * np.sin(theta) - self.rho
        sigma = fwhm * FWHM_TO_SIGMA

        return self._blurred_box(distance, sigma) * self.peak_signal

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
