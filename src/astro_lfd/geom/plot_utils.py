from __future__ import annotations

__all__ = ["plot_camera", "plot_line", "plot_line_segment"]

from typing import TYPE_CHECKING

from lsst.afw.cameraGeom import Camera, Detector, FOCAL_PLANE, PIXELS
from matplotlib.patches import Polygon

if TYPE_CHECKING:
    from lsst.afw.geom import TransformPoint2Point2
    from matplotlib.axes import Axes

    from .line import Line2D, LineSegment2D


def plot_camera(ax: Axes, camera: Camera, add_labels: bool = False, **kwargs) -> None:
    """Plot detectors comprising a camera.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        A (sub-)plot in a figure.
    camera : `lsst.afw.cameraGeom.Camera`
        The camera containing the detectors to plot.
    set_label : `bool`, optional
        Add detector name labels to the plot if `True` (`False`, by default).
    **kwargs
        Additional keyword arguments passed to `matplotlib.patches.Polygon`.
    """
    for detector in camera:
        corners = detector.getCorners(FOCAL_PLANE)
        patch = Polygon(
            [(p.getX(), p.getY()) for p in corners],
            closed=True,
            fill=False,
            **kwargs,
        )
        ax.add_patch(patch)

        if add_labels:
            center = detector.getOrientation().getFpPosition()
            ax.text(
                center.getX(),
                center.getY(),
                detector.getName(),
                ha="center",
                va="center",
            )

    ax.autoscale()
    ax.set_aspect("equal")


def plot_line_segment(
    ax: Axes,
    line_segment: LineSegment2D,
    transform: TransformPoint2ToPoint2 | None,
    **kwargs,
) -> None:
    """Plot a `LineSegment2D`.

    Parameters
    ----------
    axes : `matplotlib.axes.Axes`
        A (sub-)plot in a figure.
    line_segment : `astro_lfd.geom.LineSegment2D`
        The line segment to plot.
    transform : `lsst.afw.geom.TransformPoint2ToPoint2`, optional
        Coordinate transform to apply to points.
    **kwargs
        Additional keyword arguments passed to `matplotlib.axes.Axes.plot`.
    """
    points = [line_segment.p0, line_segment.p1]
    if transform:
        points = transform.applyForward(points)

    ax.plot([p.x for p in points], [p.y for p in points], **kwargs)


def plot_line(
    ax: Axes,
    line: Line2D,
    detector: Detector,
    use_camera_coordinates: bool = False,
    set_limits: bool = True,
    **kwargs,
) -> None:
    """Plot a `Line2D` clipped to a bounding box.

    Parameters
    ----------
    axes : `matplotlib.axes.Axes`
        A (sub-)plot in a figure.
    line : `astro_lfd.geom.Line2D`
        The line to plot.
    detector : `lsst.afw.cameraGeom.Detector`
        The detector to derive the bounding box from.
    use_camera_coordinates : `bool`, optional
        Plot using camera coordinates if `True` (`False`, by default).
    set_limits : `bool`, optional
        Set axis limits to match the bounding box extent if `True`.
    **kwargs
        Additional keyword arguments passed to `matplotlib.axes.Axes.plot`.
    """
    box = detector.getBBox()
    line_segment = line.clipped_to(box)

    transform = detector.getTransform(PIXELS, FOCAL_PLANE) if use_camera_coordinates else None
    plot_line_segment(ax, line_segment, transform=transform, **kwargs)

    if set_limits and not use_camera_coordinates:
        ax.set_ylim(box.beginY, box.endY)
        ax.set_xlim(box.beginX, box.endX)
