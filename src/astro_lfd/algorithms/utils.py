from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Concatenate, ParamSpec, TypeVar

from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
import numpy as np

import lsst.afw.image as afwImage

__all__ = ["get_pixel_mask", "binary_dilation", "timed"]


P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")


def get_pixel_mask(mask: afwImage.Mask, mask_plane: str | list[str]) -> NDArray[np.bool_]:
    """Get the pixel mask array corresponding to the named mask planes.

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


def timed(step: str):
    def decorator[**P, R](
        func: Callable[Concatenate[S, P], R],
    ) -> Callable[Concatenate[S, P], R]:
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
