from __future__ import annotations

__all__ = ["Line2D", "LineGeometry2D", "LineSegment2D"]

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import lsst.geom as geom
import numpy as np
from numpy.typing import ArrayLike, NDArray


class LineGeometry2D(ABC):
    """Abstract base class for transformable line geometries."""

    _TRANSFORM_BASELINE: float = 10.0
    """The distance between two defining points for a line (`float`)."""

    @classmethod
    @abstractmethod
    def from_points(cls, p0: geom.Point2D, p1: geom.Point2D) -> Self:
        """Construct the line geometry from two defining points.

        Parameters
        ----------
        p0, p1 : `lsst.geom.Point2D`
            The two defining points.

        Returns
        -------
        line_geom : `astro_lfd.geom.LineGeometry2D`
            The line geometry defined by the two points.
        """
        ...

    @abstractmethod
    def as_line(self) -> Line2D:
        """Get the line representation of the line geometry.

        Returns
        -------
        line : `astro_lfd.geom.Line2D`
            The line representation.
        """
        ...

    @abstractmethod
    def at(self, s: float) -> geom.Point2D:
        """Evaluate the line geometry at the along-line coordinate.

        Parameters
        ----------
        s : `float`
            The (signed) coordinate along the line direction.

        Returns
        -------
        point : `lsst.geom.Point2D`
            The point on the line at the along-line coordinate.
        """
        ...

    @abstractmethod
    def contains(self, point: geom.Point2D, atol: float = 1e-12) -> bool:
        """Return `True` if point lies on the line geometry.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            The point to test.
        atol : `float`, optional
            The maximum allowable distance, in pixels, between the point and
            the line geometry (1e-12, by default).

        Returns
        -------
        does_contain : `bool`
            `True` if the point lies on the line geometry, `False` if not.
        """
        ...

    def clipped_to(self, box: geom.Box2D | geom.Box2I) -> LineSegment2D | None:
        """Return the portion of the line geometry inside a box.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box boundary to clip the line geometry to.

        Returns
        -------
        line_segment : `astro_lfd.geom.LineSegment2D`
            The portion of the line geometry inside the box, or `None` if the
            geometry does not intersect the box.
        """
        interval = self._interval_in_box(box)
        if interval is None:
            return None

        return LineSegment2D(self.as_line(), interval=interval)

    @abstractmethod
    def boundary_intersections(
        self,
        box: geom.Box2I | geom.Box2D,
        atol: float = 1e-12,
    ) -> list[geom.Point2D]:
        """Return the line geometry intersection points with a box boundary.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box boundary to intersect the line geometry with.
        atol : `float`, optional
            The minimum allowable difference, in pixels, of the direction
            vector components from zero (1e-12, by default).

        Returns
        -------
        points : `list` [`lsst.geom.Point2D`]
            The list of intersection points (empty, if none exist).
        """
        ...

    def transformed(self, transform: Any) -> Self:
        """Apply a geometric transformation.

        The supplied transform maps points expressed in the current coordinate
        system into points in the target coordinate system.

        Parameters
        ----------
        transform : `lsst.geom.AffineTransform` or \
                    `lsst.afw.geom.TransformPoint2ToPoint2`
            Transform that maps points from the current coordinate system into
            the target coordinate system.

        Returns
        -------
        transformed : `astro_lfd.geom.LineGeometry2D`
            A new line geometry in the target coordinate system.
        """
        p0, p1 = self._defining_points()

        p0_t = _apply_transform(transform, p0)
        p1_t = _apply_transform(transform, p1)

        return type(self).from_points(p0_t, p1_t)

    def rotated(self, angle: geom.Angle) -> Self:
        """Apply a rotational transformation.

        Parameters
        ----------
        angle : `lsst.geom.Angle`
            The angle to rotate by.

        Returns
        -------
        transformed : `astro_lfd.geom.LineGeometry2D`
            The transformed line geometry.
        """
        return self.transformed(geom.AffineTransform.makeRotation(angle))

    def scaled(self, factor: float) -> Self:
        """Apply a scaling transformation.

        Parameters
        ----------
        factor : `float`
            The factor to scale by.

        Returns
        -------
        transformed : `astro_lfd.geom.LineGeometry2D`
            The transformed line geometry.
        """
        return self.transformed(geom.AffineTransform.makeScaling(factor))

    def translated(self, offset: geom.Extent2D) -> Self:
        """Apply a translation transformation.

        Parameters
        ----------
        offset : `lsst.geom.Extent2D`
            The offset to translate by.

        Returns
        -------
        transformed : `astro_lfd.geom.LineGeometry2D`
            The transformed line geometry.
        """
        return self.transformed(geom.AffineTransform.makeTranslation(offset))

    @abstractmethod
    def _defining_points(self) -> tuple[geom.Point2D, geom.Point2D]:
        """Return the two points defining the line geometry.

        Returns
        -------
        p0, p1 : `lsst.geom.Point2D`
            The two defining points.
        """
        ...

    @abstractmethod
    def _interval_in_box(self, box: geom.Box2I | geom.Box2D) -> geom.IntervalD | None:
        """Return the valid parameter interval in a box boundary.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box boundary to constrain the line geometry interval within.

        Returns
        -------
        interval : `lsst.geom.IntervalD`
            The parameter interval in the box.
        """
        ...


class Line2D(LineGeometry2D):
    """A line geometric primitive."""

    def __init__(self, rho: float, theta: geom.Angle):
        rho, theta = _canonicalize(rho, theta)
        self._rho = rho
        self._theta = theta

    @classmethod
    def from_points(cls, p0: geom.Point2D, p1: geom.Point2D) -> Self:
        """Create a `Line2D` instance from two defining points.

        Parameters
        ----------
        p0, p1 : `lsst.geom.Point2D`
            The two defining points.

        Returns
        -------
        line : `astro_lfd.geom.Line2D`
            An instance of `Line2D` defined by the two points.
        """
        return cls.from_point_and_direction(p0, p1 - p0)

    @classmethod
    def from_point_and_direction(cls, point: geom.Point2D, direction: geom.Extent2D) -> Self:
        """Create a `Line2D` instance from a point and a direction vector.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            A point on the line.
        direction : `lsst.geom.Extent2D`
            The direction vector of the line.

        Returns
        -------
        line : `astro_lfd.geom.Line2D`
            An instance of `Line2D` defined by the point and direction.

        Raises
        ------
        ValueError
            Raised if the direction vector is zero.
        """
        if (norm := direction.computeNorm()) == 0:
            raise ValueError("direction vector must be non-zero")

        direction = direction / norm
        normal = geom.Extent2D(-direction.y, direction.x)

        rho = _dot(geom.Extent2D(point), normal)
        theta = math.atan2(normal.y, normal.x) * geom.radians

        return cls(rho, theta)

    @property
    def direction(self) -> geom.Extent2D:
        """The line direction vector (`lsst.geom.Extent2D`)."""
        t = self.theta.asRadians()
        return geom.Extent2D(-math.sin(t), math.cos(t))

    @property
    def normal(self) -> geom.Extent2D:
        """The line normal vector (`lsst.geom.Extent2D`)."""
        t = self.theta.asRadians()
        return geom.Extent2D(math.cos(t), math.sin(t))

    @property
    def origin(self) -> geom.Point2D:
        """The point on the line closest to the origin (`lsst.geom.Point2D`)."""
        return geom.Point2D(self.normal * self.rho)

    @property
    def rho(self) -> float:
        """The signed perpendicular distance from the origin to the line
        (`float`).
        """
        return self._rho

    @property
    def theta(self) -> geom.Angle:
        """The angle of the line normal vector (`lsst.geom.Angle`)."""
        return self._theta

    def as_line(self) -> Line2D:
        """Get the line representation of the line geometry.

        Returns
        -------
        line : `astro_lfd.geom.Line2D`
            The line representation.
        """
        return self

    def at(self, s: float) -> geom.Point2D:
        """Evaluate the line geometry at the along-line coordinate.

        Parameters
        ----------
        s : `float`
            The (signed) coordinate along the line direction.

        Returns
        -------
        point : `lsst.geom.Point2D`
            The point on the line at the along-line coordinate.
        """
        return self.origin + self.direction * s

    def along_coordinate(self, point: geom.Point2D) -> float:
        """Return the signed coordinate along the line direction.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Point to project along the line direction vector.

        Returns
        -------
        s : `float`
            The signed coordinate along the line direction relative to the
            point closest to the origin.
        """
        delta = point - self.origin
        return _dot(delta, self.direction)

    def normal_coordinate(self, point: geom.Point2D) -> float:
        """Return the signed coordinate along the line normal.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Point to project along the line normal vector.

        Returns
        -------
        r : `float`
            The signed coordinate along the line normal relative to the point
            closest to the origin.
        """
        delta = point - self.origin
        return _dot(delta, self.normal)

    def contains(self, point: geom.Point2D, atol: float = 1e-12) -> bool:
        """Return `True` if point lies on the line geometry.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            The point to test.
        atol : `float`, optional
            The maximum allowable distance, in pixels, between the point and
            the line geometry (1e-12, by default).

        Returns
        -------
        does_contain : `bool`
            `True` if the point lies on the line geometry, `False` if not.
        """
        return abs(self.normal_coordinate(point)) <= atol

    def boundary_intersections(
        self,
        box: geom.Box2I | geom.Box2D,
        atol: float = 1e-12,
    ) -> list[geom.Point2D]:
        """Return the line geometry intersection points with a box boundary.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box boundary to intersect the line geometry with.
        atol : `float`, optional
            The minimum allowable difference, in pixels, of the direction
            vector components from zero (1e-12, by default).

        Returns
        -------
        points : `list` [`lsst.geom.Point2D`]
            The list of boundary intersection points (empty, if none exist).
        """
        o = self.origin
        d = self.direction

        xmin = box.minX
        xmax = box.maxX
        ymin = box.minY
        ymax = box.maxY

        points: list[geom.Point2D] = []
        if abs(d.x) > atol:
            for x in (xmin, xmax):
                p = self.at((x - o.x) / d.x)
                if ymin - atol <= p.y <= ymax + atol:
                    points.append(p)

        if abs(d.y) > atol:
            for y in (ymin, ymax):
                p = self.at((y - o.y) / d.y)
                if xmin - atol <= p.x <= xmax + atol:
                    points.append(p)

        unique: list[geom.Point2D] = []
        for p in points:
            for q in unique:
                if p.distanceSquared(q) <= atol * atol:
                    break
            else:
                unique.append(p)

        return unique

    def _defining_points(self) -> tuple[geom.Point2D, geom.Point2D]:
        """Return two points defining the line.

        Returns
        -------
        p0, p1 : `lsst.geom.Point2D`
            The two defining points.
        """
        return self.origin, self.at(self._TRANSFORM_BASELINE)

    def _interval_in_box(self, box: geom.Box2I | geom.Box2D) -> geom.IntervalD | None:
        """Return the valid parameter interval in a box boundary.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box boundary to constrain the line geometry interval within.

        Returns
        -------
        interval : `lsst.geom.IntervalD`
            The parameter interval in the box.
        """
        return _line_box_interval(self, box)


class LineSegment2D(LineGeometry2D):
    """A line segment geometric primitive."""

    def __init__(self, line: Line2D, interval: geom.IntervalD):
        self._line = line
        self._interval = interval

    @classmethod
    def from_center_length(cls, line: Line2D, s_center: float, length: float) -> Self:
        h = 0.5 * length
        interval = geom.IntervalD.fromSpannedPoints((s_center - h, s_center + h))
        return cls(line, interval)

    @classmethod
    def from_points(cls, p0: geom.Point2D, p1: geom.Point2D) -> Self:
        """Create a `LineSegment2D` instance from two defining points.

        Parameters
        ----------
        p0, p1 : `lsst.geom.Point2D`
            The two defining points.

        Returns
        -------
        line : `astro_lfd.geom.LineSegment2D`
            An instance of `LineSegment2D` defined by the two points.
        """
        line = Line2D.from_points(p0, p1)
        interval = geom.IntervalD.fromSpannedPoints([line.along_coordinate(p0), line.along_coordinate(p1)])

        return cls(line, interval)

    @property
    def interval(self) -> geom.IntervalD:
        """The parameter interval of the line segment (`geom.IntervalD`)."""
        return self._interval

    @property
    def line(self) -> Line2D:
        """The infinite line representation (`Line2D`)."""
        return self._line

    @property
    def rho(self) -> float:
        """The signed perpendicular distance from the origin to the line
        (`float`).
        """
        return self._line.rho

    @property
    def theta(self) -> geom.Angle:
        """The angle of the line normal vector (`lsst.geom.Angle`)."""
        return self._line.theta

    @property
    def length(self) -> float:
        """The length of the line (`float`)."""
        return self.interval.size

    @property
    def s_center(self) -> float:
        """The center along-line coordinate of the line segment (`float`)."""
        return self.interval.center

    @property
    def s_max(self) -> float:
        """The maximum along-line coordinate of the line segment (`float`)."""
        return self.interval.max

    @property
    def s_min(self) -> float:
        """The minimum along-line coordinate of the line segment (`float`)."""
        return self.interval.min

    @property
    def center(self) -> geom.Point2D:
        """The center point of the line segment (`geom.Point2D`)."""
        return self.at(self.s_center)

    @property
    def p0(self) -> geom.Point2D:
        """The start point of the line segment (`geom.Point2D`)."""
        return self.at(self.s_min)

    @property
    def p1(self) -> geom.Point2D:
        """The end point of the line segment (`geom.Point2D`)."""
        return self.at(self.s_max)

    def as_line(self) -> Line2D:
        """Get the line representation of the line geometry.

        Returns
        -------
        line : `astro_lfd.geom.Line2D`
            The line representation.
        """
        return self.line

    def at(self, s: float) -> geom.Point2D:
        """Evaluate the line geometry at the along-line coordinate.

        Parameters
        ----------
        s : `float`
            The signed coordinate along the line direction.

        Returns
        -------
        point : `lsst.geom.Point2D`
            The point on the line at the along-line coordinate.
        """
        return self.line.at(s)

    def contains(self, point: geom.Point2D, atol: float = 1e-12) -> bool:
        """Return `True` if point lies on the line geometry.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            The point to test.
        atol : `float`, optional
            The maximum allowable distance, in pixels, between the point and
            the line geometry (1e-12, by default).

        Returns
        -------
        does_contain : `bool`
            `True` if the point lies on the line geometry, `False` if not.
        """
        if not self.line.contains(point, atol=atol):
            return False

        s = self.line.along_coordinate(point)
        return (self.interval.min - atol) <= s <= (self.interval.max + atol)

    def boundary_intersections(
        self,
        box: geom.Box2I | geom.Box2D,
        atol: float = 1e-12,
    ) -> list[geom.Point2D]:
        """Return the line geometry intersection points with a box boundary.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box boundary to intersect the line geometry with.
        atol : `float`, optional
            The minimum allowable difference, in pixels, of the direction
            vector components from zero (1e-12, by default).

        Returns
        -------
        points : `list` [`lsst.geom.Point2D`]
            The list of boundary intersection points (empty, if none exist).
        """
        return [p for p in self.line.boundary_intersections(box, atol=atol) if self.contains(p, atol=atol)]

    def _defining_points(self) -> tuple[geom.Point2D, geom.Point2D]:
        """Return two points defining the line.

        Returns
        -------
        p0, p1 : `lsst.geom.Point2D`
            The two defining points.
        """
        return self.p0, self.p1

    def _interval_in_box(self, box: geom.Box2I | geom.Box2D) -> geom.IntervalD | None:
        """Return the valid parameter interval in a box boundary.

        Parameters
        ----------
        box : `lsst.geom.Box2I` or `lsst.geom.Box2D`
            The box boundary to constrain the line geometry interval within.

        Returns
        -------
        interval : `lsst.geom.IntervalD`
            The parameter interval in the box.
        """
        box_interval = _line_box_interval(self.line, box)

        if box_interval is None:
            return None

        smin = max(self.interval.min, box_interval.min)
        smax = min(self.interval.max, box_interval.max)

        if smin > smax:
            return None

        return geom.IntervalD.fromSpannedPoints([smin, smax])


def _apply_transform(transform: Any, point: geom.Point2D) -> geom.Point2D:
    """Apply a transform to a point.

    Apply a transformation to a point either by calling the applyForward
    method of the object or calling the object directly.

    Parameters
    ----------
    transform:
        The object the encapsulates the transformation.
    point: `lsst.geom.Point2D`
        The point to transform.

    Returns
    -------
    transformed : `lsst.geom.Point2D`
        The transformed point.

    Raises
    ------
    TypeError
        Raised if ``transform`` does not have a valid callable interface.
    """
    if hasattr(transform, "applyForward"):
        return transform.applyForward(point)

    if callable(transform):
        return transform(point)

    raise TypeError("transform is invalid callable or object")


def _canonicalize(rho: float, theta: geom.Angle) -> tuple[float, geom.Angle]:
    """Convert line parameters to a canonical Hesse normal form.

    Parameters
    ----------
    rho : `float`
        Signed perpendicular distance from the origin to the line.
    theta : `float`
        Angle of the line normal vector.

    Returns
    -------
    canonical_rho : `float`
        Canonical signed perpendicular distance.
    canonical_theta : `lsst.geom.Angle`
        Canonical normal angle in the interval [0, pi).
    """
    theta_rad = theta.asRadians() % (2 * math.pi)

    if theta_rad >= math.pi:
        theta_rad -= math.pi
        rho = -rho

    return rho, theta_rad * geom.radians


def _dot(v0: geom.Extent2D, v1: geom.Extent2D) -> float:
    """Compute the Euclidean dot product of two vectors.

    Parameters
    ----------
    v0 : `lsst.geom.Extent2D`
        The first vector.
    v1 : `lsst.geom.Extent2D`
        The second vector.

    Returns
    -------
    dot_product : `float`
        The dot product of the two vectors.
    """
    return v0.x * v1.x + v0.y * v1.y


def _line_box_interval(line: Line2D, box: geom.Box2D | geom.Box2I) -> geom.IntervalD | None:
    """Compute the line parameter interval inside a box.

    Uses slab intersection in parametric form.
    """
    origin = line.origin
    direction = line.direction

    xmin = box.minX
    xmax = box.maxX
    ymin = box.minY
    ymax = box.maxY

    smin = -math.inf
    smax = math.inf

    def clip_axis(p0: float, dp: float, lo: float, hi: float) -> tuple[float, float] | None:
        if abs(dp) < 1e-15:
            if p0 < lo or p0 > hi:
                return None

            return -math.inf, math.inf

        s0 = (lo - p0) / dp
        s1 = (hi - p0) / dp

        return min(s0, s1), max(s0, s1)

    if (x_interval := clip_axis(origin.x, direction.x, xmin, xmax)) is None:
        return None

    if (y_interval := clip_axis(origin.y, direction.y, ymin, ymax)) is None:
        return None

    smin = max(smin, x_interval[0], y_interval[0])
    smax = min(smax, x_interval[1], y_interval[1])

    if smin > smax:
        return None

    return geom.IntervalD.fromSpannedPoints([smin, smax])
