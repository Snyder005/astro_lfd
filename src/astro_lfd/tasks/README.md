# tasks

## Overview

Streak analysis and post-processing tasks. Tasks that can be incorporated into
science pipelines are defined here.

## Architecture

**LFD hierarchy.** An individual linear feature detector `Task` to be used in 
`StreakAnalysisTask` is selected by setting the `detection_algorithm` 
configurable parameter in `StreakAnalysisConfig`. New detector implementations 
are added as a new `ConfigurableField`, a key-value pair in the list of 
`allowed` values in `detection_algorithm`, and initialized via `makeSubtask()`
in the `StreakAnalysisTask` constructor. 

`WriteStreakCatalogTask` is a pipeline task meant to bridge the gap between the
existing `ArrowNumpyDict` and the new `SourceCatalog` storage class for streak
detection results. It is used to write a streak `dict` to `SourceCatalog`
format. The streak parameters are recorded in image-array coordinates centered
on the image, each line is mapped into absolute pixel coordinates by
translating from the image center; this matches the conventions of the new LFD
tasks. There is no way to separate individual mask contributions from the
"STREAK" mask plane, so the per-streak footprint is left unset.

`StreakAnalysisTask` is a pipeline task used to detect streaks in a difference
image. This is intended to isolate the streak detection algorithms from the
existing difference image analysis pipelines for testing and development of
linear feature detectors, for which it allows selection via a configurable
parameter.
