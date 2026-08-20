import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

if TYPE_CHECKING:
    import lsst.geom as geom

FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass
class Streak:
    """Streak geometry and brightness in Hesse normal form.

    The streak is a line in Hesse normal form with a top-hat cross-sectional
    profile. By default the streak extends infinitely; supplying ``length``
    constrains it to a finite segment centered at ``s_center`` along the line.

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
    length : `float` or None, optional
        The top-hat full length of the streak along the line, in pixels. If
        `None` (default), the streak extends infinitely along the line.
    s_center : `float`, optional
        The along-line coordinate of the streak center, in pixels (0.0, by
        default). The along-line coordinate is measured from the point on the
        line closest to the origin. Only meaningful when ``length`` is set.
    """

    rho: float
    theta: float
    peak_signal: float
    width: float
    length: float | None = None
    s_center: float = 0.0

    @classmethod
    def from_center_length(
        cls,
        center: "geom.Point2D | tuple[float, float]",
        theta: float,
        length: float,
        peak_signal: float,
        width: float,
    ) -> Self:
        """Create a `Streak` from a center point, orientation, and length.

        This is the intuitive PIXEL-frame constructor: the streak is placed by
        its center point rather than its Hesse parameters. The center is
        projected onto the line to recover ``rho`` and the along-line center
        ``s_center``.

        Parameters
        ----------
        center : `lsst.geom.Point2D` or `tuple` [`float`, `float`]
            The center point of the streak, in PIXEL coordinates ``(x, y)``.
        theta : `float`
            The angle of the line normal vector, in degrees.
        length : `float`
            The top-hat full length of the streak along the line, in pixels.
        peak_signal : `float`
            The signal at the line center.
        width : `float`
            The top-hat full width of the streak, in pixels.

        Returns
        -------
        streak : `Streak`
            A finite-length streak centered on ``center``.

        Notes
        -----
        This constructor requires the LSST stack (`lsst.geom`), imported
        lazily so that the module and `get_signal` remain stack-independent.
        """
        import lsst.geom as geom

        from ..geom import Line2D

        if not isinstance(center, geom.Point2D):
            center = geom.Point2D(center[0], center[1])

        theta_rad = math.radians(theta)
        direction = geom.Extent2D(-math.sin(theta_rad), math.cos(theta_rad))
        line = Line2D.from_point_and_direction(center, direction)

        return cls(
            rho=line.rho,
            theta=line.theta.asDegrees(),
            peak_signal=peak_signal,
            width=width,
            length=length,
            s_center=line.along_coordinate(center),
        )

    def get_signal(self, shape: tuple[int, int], fwhm: float | None = None) -> NDArray[np.float64]:
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
            transverse = self._box(distance, self.width / 2)
        else:
            sigma = fwhm * FWHM_TO_SIGMA
            transverse = self._blurred_box(distance, self.width / 2, sigma)

        if self.length is None:
            return transverse * self.peak_signal

        # Along-line (tangent) coordinate in the same PIXEL frame, gated by the
        # finite length. Ends taper with the same PSF sigma as the sides.
        s = -gx * np.sin(theta) + gy * np.cos(theta)
        if fwhm is None:
            longitudinal = self._box(s, self.length / 2, center=self.s_center)
        else:
            longitudinal = self._blurred_box(s, self.length / 2, sigma, center=self.s_center)

        return transverse * longitudinal * self.peak_signal

    def _box(
        self,
        coord: NDArray[np.float64],
        half_extent: float,
        center: float = 0.0,
    ) -> NDArray[np.float64]:
        """Top-hat profile along one axis, peak-normalized to 1.

        Parameters
        ----------
        coord : `numpy.ndarray`
            The signed coordinate along the axis, in pixels.
        half_extent : `float`
            The half-extent of the top-hat, in pixels.
        center : `float`, optional
            The center of the top-hat along the axis, in pixels (0.0, by
            default).

        Returns
        -------
        normalized_profile : `numpy.ndarray`
            Normalized profile values at each coordinate. Equal to 1 inside the
            top-hat half-extent.
        """
        return (np.abs(coord - center) <= half_extent).astype(np.float64)

    def _blurred_box(
        self,
        coord: NDArray[np.float64],
        half_extent: float,
        sigma: float,
        center: float = 0.0,
    ) -> NDArray[np.float64]:
        """Top-hat convolved with a Gaussian along one axis, peak-normalized
        to 1.

        Parameters
        ----------
        coord : `numpy.ndarray`
            The signed coordinate along the axis, in pixels.
        half_extent : `float`
            The half-extent of the top-hat, in pixels.
        sigma : `float`
            Gaussian sigma, in pixels.
        center : `float`, optional
            The center of the top-hat along the axis, in pixels (0.0, by
            default).

        Returns
        -------
        normalized_profile : `numpy.ndarray`
            Normalized profile values at each coordinate. Equal to 1 at the
            center of the top-hat.
        """
        d = coord - center
        y = 0.5 * (
            erf((d + half_extent) / (np.sqrt(2) * sigma)) - erf((d - half_extent) / (np.sqrt(2) * sigma))
        )
        return y / erf(half_extent / (np.sqrt(2) * sigma))
