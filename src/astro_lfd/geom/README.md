# line

## Overview

Line and line segment geometric primitives. This is where the geometric
representation of detected linear features is defined.

## Architecture

**LineGeometry2D hierarchy.** `LineGeometry2D` is an ABC that defines geometric
primitives that transform like lines.

**Vectorized coordinates.** `Line2D.normal_coordinate_array` and
`along_coordinate_array` are numpy-broadcasting forms of `normal_coordinate`
and `along_coordinate`, for evaluating line distances over pixel grids
without building `Point2D` objects.
