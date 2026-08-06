from functools import wraps
from time import perf_counter

from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

import lsst.afw.image as afwImage


__all__ = ["get_pixel_mask", "binary_dilation", "timed"]


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
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            t0 = perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                dt = perf_counter() - t0

                if not hasattr(self, "timings"):
                    self.timings = {}

                self.timings[step] = dt

                if hasattr(self, "log"):
                    self.log.info("%s took %.3f s", stage, dt)

        return wrapper
    return decorator
