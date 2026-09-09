# Pipeline Definitions

This directory contains pipeline definition YAML files which are used for
detecting streaks in astronomical images  with the LSST Science Pipelines.

The pipelines defined here come in three flavors: camera-specific (within named
directories), camera-agnostic (top-level, if any), and building-block
ingredients (within the [\_ingredients](_ingredients) directory). Pipelines
within the ingredients directory are meant to be imported by other pipelines,
and are not intended to be used directly by end-users.
