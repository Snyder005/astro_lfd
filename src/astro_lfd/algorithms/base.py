__all__ = ["binary_dilation", "get_pixel_mask", "HasTimings", "timed"]

from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Concatenate, Protocol

import lsst.afw.image as afwImage
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt


class HasTimings(Protocol):
    """A protocol defining structural behavior for timings.

    Any class implementing this protocol must have an appropriate
    ``self.timings`` parameter.
    """

    timings: dict[str, float]


def get_pixel_mask(mask: afwImage.Mask, mask_plane: str | list[str]) -> NDArray[np.bool_]:
    """Get the binary array corresponding to the named mask planes.

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


def timed[**P, R, S: HasTimings](
    step: str,
) -> Callable[[Callable[Concatenate[S, P], R]], Callable[Concatenate[S, P], R]]:
    """Decorate a method to record its execution time.

    The elapsed time is stored in ``self.timings[step]``. The timing is
    recorded even if the decorated method raises an exception.

    Parameters
    ----------
    step: `str`
        Key under which to store the elapsed time in ``self.timings``.

    Returns
    -------
    decorator : callable
        A decorator that wraps the method while preserving its parameters and
        return type.
    """

    def decorator(func: Callable[Concatenate[S, P], R]) -> Callable[Concatenate[S, P], R]:
        @wraps(func)
        def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
            t0 = perf_counter()
            try:
                return func(self, *args, **kwargs)

            finally:
                dt = perf_counter() - t0
                self.timings[step] = dt

        return wrapper

    return decorator
