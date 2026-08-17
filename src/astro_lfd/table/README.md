# table

## Overview

Adapter exposing streak line geometry on a source record. This is the bridge 
between the LSST measurement catalog to the LFD line primitives.

## Architecture

`StreakAdapter` wraps a single `lsst.afw.table.SourceRecord` and presents its
``line_*`` fields as `~astro_lfd.geom.line.Line2D` / `LineSegment2D` geometry,
bridging the LSST measurement catalog to the LFD line primitives.
