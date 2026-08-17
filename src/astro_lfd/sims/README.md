# sims

## Overview

The simulation layer provides tools for generating an image containing a single
streak, plus its variance and mask planes.

## Architecture

`SimulatedImage` represents a simulated image, variance, and mask plane as a 
light-weight dataclass. Three constructors are provided: `__init__` (image, 
variance, and mask plane plus metadata), `simulate_exposure` (from photometric
parameters and a streak), and `load_npz` (by filename). Compatibility with the
LSST Science Pipelines is provided by `to_exposure()`, outputs the simulated
image as an `lsst.afw.image.ExposureF`. 

`Streak` holds the orientation and brightness of the streak; `_blurred_box()`
implements a default profile of a top-hat convolved with a Gaussian. The method
`get_signal()` accepts a 2-D array shape and calculates the signal at each
pixel given its distance from the streak.

## Design Decisions

The radiometric simulation is always done in **electrons** because the noise
model requires it: sky and streak shot noise are Poisson (valid only on
electron counts) and read noise is quoted in electrons.  The final image and
variance are then converted to the requested output ``unit`` via a single
scalar calibration factor ``calib`` (electrons per output unit).

I/O is provided in two formats: compact ``.npz`` (no LSST dependency) for numpy
development, and ``lsst.afw.image.ExposureF`` FITS (LSST Science Pipelines
compatible).  Converters between the two are included.
