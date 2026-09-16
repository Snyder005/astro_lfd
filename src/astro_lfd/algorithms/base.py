__all__ = ["get_line_mask", "get_pixel_mask", "HasTimings", "timed"]

from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Concatenate, Protocol

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from ..geom.line import Line2D


class HasTimings(Protocol):
    """A protocol defining structural behavior for timings.

    Any class implementing this protocol must have an appropriate
    ``self.timings`` parameter.
    """

    timings: dict[str, float]


def get_line_mask(line: Line2D, shape: tuple[int, int], width: float) -> NDArray[np.bool_]:
    """Get the binary array corresponding to a region around a line.

    Parameters
    ----------
    line : `astro_lfd.geom.Line2D`
        The line used to define the mask.
    shape : `tuple` [`int`]
        The shape of the array.
    width : `float`
        The width of the region centered around the line.

    Returns
    -------
    line_mask : `numpy.ndarray`, (Ny, Nx)
        Boolean array, `True` the in region around the line.
    """
    rho = line.rho
    theta = line.theta.asRadians()

    Y, X = np.ogrid[:shape[0], :shape[1]]
    mask = np.abs((X * np.cos(theta) + Y * np.sin(theta)) - rho) < width / 2.0

    return mask


def get_pixel_mask(mask: afwImage.Mask,
    mask_plane: str | list[str],
    dilation: int = 0,
) -> NDArray[np.bool_]:
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
    ignore_mask = mask.clone()
    bitmask = mask.getPlaneBitMask(mask_plane)

    if dilation > 0:
        bbox = ignore_mask.getBBox()
        for sset in afwGeom.SpanSet.fromMask(mask, bitmask).split():
            dilated_sset = sset.dilated(dilation)
            dilated_sset.clippedTo(bbox).setMask(ignore_mask, ignore_mask.getPlaneBitMask("BAD"))

    return (ignore_mask.array & bitmask) > 0


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
