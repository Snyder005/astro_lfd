"""Unit tests for :mod:`astro_lfd.table.streakAdapter`.

Covers the minimal schema and the line-segment round-trip through a real afw
`SourceRecord`.  The whole module is skipped when the LSST afw table stack is
unavailable so the core suite runs without it.
"""

from __future__ import annotations

import math

import pytest

afwTable = pytest.importorskip("lsst.afw.table", reason="LSST afw.table not available")
geom = pytest.importorskip("lsst.geom", reason="LSST geom not available")

from astro_lfd.geom.line import Line2D, LineSegment2D  # noqa: E402
from astro_lfd.table.streakAdapter import StreakAdapter  # noqa: E402


def _make_adapter() -> StreakAdapter:
    """Build a StreakAdapter backed by a fresh, empty source record."""
    schema = StreakAdapter.makeMinimalSchema()
    table = afwTable.SourceTable.make(schema)
    catalog = afwTable.SourceCatalog(table)
    return StreakAdapter(catalog.addNew())


# --- schema -----------------------------------------------------------------
def test_minimal_schema_has_line_fields() -> None:
    """The minimal schema exposes the streak ``line_*`` fields."""
    schema = StreakAdapter.makeMinimalSchema()
    names = set(schema.getNames())
    for field in ("line_rho", "line_theta", "line_u_center", "line_length"):
        assert field in names


# --- round-trip -------------------------------------------------------------
def test_line_segment_roundtrip() -> None:
    """set/getLineSegment preserves rho, theta, center, and length."""
    adapter = _make_adapter()
    line = Line2D(rho=12.5, theta=0.5 * geom.radians)
    segment = LineSegment2D.from_center_length(line=line, u_center=4.0, length=20.0)

    adapter.setLineSegment(segment)
    out = adapter.getLineSegment()

    assert out.rho == pytest.approx(segment.rho)
    assert out.theta.asRadians() == pytest.approx(segment.theta.asRadians())
    assert out.u_center == pytest.approx(4.0)
    assert out.length == pytest.approx(20.0)


def test_getline_matches_stored_fields() -> None:
    """getLine reflects the stored rho/theta fields."""
    adapter = _make_adapter()
    adapter["line_rho"] = 7.0
    adapter["line_theta"] = 0.25 * math.pi * geom.radians
    line = adapter.getLine()
    assert line.rho == pytest.approx(7.0)
    assert line.theta.asRadians() == pytest.approx(0.25 * math.pi)


# --- repr -------------------------------------------------------------------
def test_repr_renders() -> None:
    """__repr__ produces a readable summary without error."""
    adapter = _make_adapter()
    segment = LineSegment2D.from_center_length(
        line=Line2D(rho=1.0, theta=0.0 * geom.radians),
        u_center=0.0,
        length=5.0,
    )
    adapter.setLineSegment(segment)
    text = repr(adapter)
    assert text.startswith("StreakAdapter(")
    assert "length=5.00" in text
