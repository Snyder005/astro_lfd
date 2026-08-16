# algorithms

## Overview

Linear feature detection tasks. Each LFD implementation follows the standard
template of pre-processing, detection, and post-processing steps.

## Architecture

**LFD template.** `HasTimings`

## Design Decisions

`KHTDetectTask` finds straight linear features (satellite streaks, and similar
signals) in an `lsst.afw.image.Exposure`.  It reproduces the detection stages of
`lsst.meas.algorithms.maskStreaks.MaskStreaksTask` -- Canny edge extraction,
`lsst.kht` line finding, recursive k-means clustering, and a Moffat line-profile
fit -- but emits its results as a `~lsst.afw.table.SourceCatalog` of canonical
line segments (via `~astro_lfd.table.streakAdapter.StreakAdapter`) instead of a
mask plane.

The profile fitter (`Line`, `LineProfile`) is imported from ``maskStreaks``
rather than reimplemented, so any difference in the fit itself is shared between
the two tasks.  This module is intended as the template for future
``astro_lfd`` detector tasks.
