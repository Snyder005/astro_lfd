# geom.line — line geometry primitives

**When relevant:** working with detected-line geometry — `rho`/`theta`
conventions, clipping a line to a detector box, fitting a segment to points, or
the afw `StreakAdapter` bridge.

**Verified:** exercised against the installed `lsst.geom` (LSST stack
`lsst-scipipe-13.0.0`), 2026-07-06.

## Convention (Hesse normal form)

A `Line2D` is stored as `(rho, theta)`:
- `theta` is the angle of the **normal** vector, canonicalized to `[0, π)`.
- `rho` is the signed perpendicular distance from the origin to the line.
- `_canonicalize` wraps `theta` into `[0, π)` and flips the sign of `rho` when
  it subtracts π — so the same geometric line always has one representation.
- `normal = (cos θ, sin θ)`, `direction = (-sin θ, cos θ)` (unit, perpendicular).

## API surface (`astro_lfd.geom.line`)

- `Line2D(rho, theta)` — `theta` is an `lsst.geom.Angle` (e.g. `0.5 * geom.radians`).
  - `.from_points(p0, p1)`, `.from_point_and_direction(point, direction)`
    (zero direction → `ValueError`).
  - `.signed_distance(point)`, `.project(point)` (coord along direction),
    `.at(s)`, `.contains(point, atol=1e-12)`.
  - `.intersections_with_box_edges(box, atol)`, `.clipped_to(box)` /
    `.intersection(box)` → `LineSegment2D | None`.
- `LineSegment2D(line, interval)` — `.from_center_length(line, u_center, length)`,
  `.from_points(p0, p1)`; `.length`, `.u_center`, `.p0`, `.p1`, `.contains`.
- `fit_line_segment_from_xy(x, y, weights=None)` → **`LineFitResult`**
  (`.line_segment`, `.rms`, `.width = 2.355 * rms`, `.aspect_ratio`). Needs ≥2
  valid weighted points. SVD of weighted, centroid-centered points; smallest
  singular vector is the normal.
- `embed_rho_theta(rho, theta, rho_tol, theta_tol)` → `(N, 3)` euclidean
  embedding for clustering nearby lines; tolerances must be `> 0`.

## Gotchas

- `fit_line_segment_from_xy` returns a `LineFitResult`, **not** a `LineSegment2D`
  — the segment is `result.line_segment`.
- `contains` is implemented via `signed_distance` (there is no `distance`
  method). An early import from `mixcoatl` called `self.distance(...)` and
  crashed — regression-tested in `tests/test_line.py`.
- **Hard dependency on `lsst.geom`.** Unlike the LFD core, this module and
  `table.streakAdapter` (which also needs `lsst.afw.table`/`lsst.afw.detection`)
  cannot import without the LSST stack. Tests guard with
  `pytest.importorskip("lsst.geom")`.

**See also:** [testdata](testdata.md)
