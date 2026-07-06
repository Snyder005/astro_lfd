"""Unit tests for :mod:`astro_lfd.geom.line`.

Covers the line/segment primitives (Hesse-normal-form construction, distance
and projection, box clipping and edge intersection), the weighted line fit, and
the rho/theta embedding.  The whole module is skipped when ``lsst.geom`` is
unavailable so the core suite runs without the LSST stack.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

geom = pytest.importorskip("lsst.geom", reason="LSST geom not available")

from astro_lfd.geom.line import (  # noqa: E402
    Line2D,
    LineFitResult,
    LineSegment2D,
    embed_rho_theta,
    fit_line_segment_from_xy,
)


# --- Line2D construction ----------------------------------------------------
def test_from_points_vertical_line() -> None:
    """A vertical line x=5 has rho=5 and a normal angle of 0 degrees."""
    line = Line2D.from_points(geom.Point2D(5.0, 0.0), geom.Point2D(5.0, 10.0))
    assert line.rho == pytest.approx(5.0)
    assert line.theta.asDegrees() == pytest.approx(0.0)


def test_from_point_and_direction_zero_raises() -> None:
    """A zero direction vector is rejected."""
    with pytest.raises(ValueError):
        Line2D.from_point_and_direction(geom.Point2D(0.0, 0.0), geom.Extent2D(0.0, 0.0))


def test_canonical_theta_in_half_open_pi() -> None:
    """theta is canonicalized into [0, pi); rho flips sign accordingly."""
    # theta = 3pi/2 (270 deg) wraps to pi/2 with a negated rho.
    line = Line2D(rho=4.0, theta=1.5 * math.pi * geom.radians)
    assert 0.0 <= line.theta.asRadians() < math.pi
    assert line.theta.asRadians() == pytest.approx(0.5 * math.pi)
    assert line.rho == pytest.approx(-4.0)


# --- geometry: normal/direction/distance/project ----------------------------
def test_normal_and_direction_orthonormal() -> None:
    """normal and direction are unit vectors and mutually perpendicular."""
    line = Line2D(rho=3.0, theta=0.7 * geom.radians)
    n, d = line.normal, line.direction
    assert math.hypot(n.x, n.y) == pytest.approx(1.0)
    assert math.hypot(d.x, d.y) == pytest.approx(1.0)
    assert n.x * d.x + n.y * d.y == pytest.approx(0.0, abs=1e-12)


def test_signed_distance_sign_and_magnitude() -> None:
    """Signed distance is +/- the perpendicular offset from the line."""
    line = Line2D.from_points(geom.Point2D(5.0, 0.0), geom.Point2D(5.0, 10.0))
    assert line.signed_distance(geom.Point2D(8.0, 3.0)) == pytest.approx(3.0)
    assert line.signed_distance(geom.Point2D(2.0, 3.0)) == pytest.approx(-3.0)


def test_contains_on_and_off_line() -> None:
    """contains uses signed_distance: True on the line, False off it.

    Regression for a bug where contains called a nonexistent ``distance``
    method and crashed on any use.
    """
    line = Line2D.from_points(geom.Point2D(5.0, 0.0), geom.Point2D(5.0, 10.0))
    assert line.contains(geom.Point2D(5.0, 42.0))
    assert not line.contains(geom.Point2D(6.0, 42.0))


def test_project_roundtrip() -> None:
    """project(at(s)) recovers s along the line direction."""
    line = Line2D(rho=2.0, theta=0.3 * geom.radians)
    for s in (-7.5, 0.0, 12.25):
        assert line.project(line.at(s)) == pytest.approx(s)


# --- intersections_with_box_edges -------------------------------------------
def test_intersections_diagonal_two_points() -> None:
    """A diagonal through the box yields exactly two deduped points."""
    line = Line2D.from_points(geom.Point2D(0.0, 0.0), geom.Point2D(10.0, 10.0))
    box = geom.Box2D(geom.Point2D(0.0, 0.0), geom.Point2D(10.0, 10.0))
    pts = line.intersections_with_box_edges(box)
    assert len(pts) == 2


def test_intersections_through_corner_not_double_counted() -> None:
    """A line through a box corner is not counted twice.

    Regression for a mis-indented for/else dedup loop that dropped all but
    the last candidate point.
    """
    # y = x passes exactly through corners (0,0) and (10,10).
    line = Line2D.from_points(geom.Point2D(0.0, 0.0), geom.Point2D(10.0, 10.0))
    box = geom.Box2D(geom.Point2D(0.0, 0.0), geom.Point2D(10.0, 10.0))
    pts = line.intersections_with_box_edges(box, atol=1e-9)
    # The two corners are distinct; each corner must appear only once.
    assert len(pts) == 2


# --- clipped_to / intersection ----------------------------------------------
def test_clipped_to_inside_box() -> None:
    """A line crossing the box clips to a segment spanning it."""
    line = Line2D.from_points(geom.Point2D(5.0, 0.0), geom.Point2D(5.0, 10.0))
    box = geom.Box2D(geom.Point2D(0.0, 0.0), geom.Point2D(10.0, 10.0))
    seg = line.clipped_to(box)
    assert seg is not None
    assert seg.length == pytest.approx(10.0)


def test_clipped_to_missing_box_returns_none() -> None:
    """A line entirely outside the box clips to None."""
    line = Line2D.from_points(geom.Point2D(50.0, 0.0), geom.Point2D(50.0, 10.0))
    box = geom.Box2D(geom.Point2D(0.0, 0.0), geom.Point2D(10.0, 10.0))
    assert line.clipped_to(box) is None


# --- LineSegment2D ----------------------------------------------------------
def test_segment_from_center_length() -> None:
    """from_center_length yields the requested length and center."""
    line = Line2D(rho=0.0, theta=0.5 * math.pi * geom.radians)
    seg = LineSegment2D.from_center_length(line=line, u_center=3.0, length=8.0)
    assert seg.length == pytest.approx(8.0)
    assert seg.u_center == pytest.approx(3.0)


def test_segment_from_points_endpoints_and_contains() -> None:
    """from_points sets endpoints; interior points are contained, exterior not."""
    p0 = geom.Point2D(0.0, 0.0)
    p1 = geom.Point2D(10.0, 0.0)
    seg = LineSegment2D.from_points(p0, p1)
    assert seg.length == pytest.approx(10.0)
    assert seg.contains(geom.Point2D(5.0, 0.0))
    # On the infinite line but beyond the segment extent.
    assert not seg.contains(geom.Point2D(20.0, 0.0))


# --- fit_line_segment_from_xy -----------------------------------------------
def test_fit_recovers_known_line() -> None:
    """A noiseless set of collinear points is fit with ~zero residual."""
    x = np.linspace(0.0, 10.0, 11)
    y = 2.0 * x + 1.0
    result = fit_line_segment_from_xy(x, y)
    assert isinstance(result, LineFitResult)
    assert result.rms == pytest.approx(0.0, abs=1e-9)
    assert result.width == pytest.approx(0.0, abs=1e-8)
    # The recovered line contains the sampled points.
    line = result.line_segment.line
    assert line.contains(geom.Point2D(x[5], y[5]), atol=1e-6)


def test_fit_too_few_points_raises() -> None:
    """Fewer than two valid weighted points is an error."""
    with pytest.raises(ValueError):
        fit_line_segment_from_xy(np.array([1.0]), np.array([2.0]))


def test_fit_mismatched_shapes_raise() -> None:
    """x and y must share a shape."""
    with pytest.raises(ValueError):
        fit_line_segment_from_xy(np.array([1.0, 2.0]), np.array([1.0]))


# --- embed_rho_theta --------------------------------------------------------
def test_embed_shape() -> None:
    """The embedding maps N (rho, theta) pairs to an (N, 3) array."""
    rho = np.array([1.0, 2.0, 3.0])
    theta = np.array([0.0, 0.1, 0.2])
    out = embed_rho_theta(rho, theta, rho_tol=1.0, theta_tol=0.1)
    assert out.shape == (3, 3)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("rho_tol, theta_tol", [(0.0, 0.1), (1.0, 0.0), (-1.0, 0.1)])
def test_embed_bad_tolerances_raise(rho_tol: float, theta_tol: float) -> None:
    """Non-positive tolerances are rejected."""
    with pytest.raises(ValueError):
        embed_rho_theta(np.array([1.0]), np.array([0.0]), rho_tol=rho_tol, theta_tol=theta_tol)
