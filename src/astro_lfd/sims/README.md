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

A streak is infinite by default. Supplying `length` (and optionally `s_center`)
constrains it to a finite segment. `get_signal()` works entirely in the
tangent–normal frame of the line: the transverse coordinate (perpendicular
distance) drives the cross-sectional width profile as before, and the
along-line coordinate drives an identical longitudinal profile gated by
`length`. Both axes share the `_box`/`_blurred_box` helpers, so a finite streak
is a soft-edged rectangle whose ends taper with the same PSF sigma as its
sides. `length=None` skips the longitudinal factor entirely, reproducing the
original infinite-line output exactly.

The `Streak.from_center_length()` constructor is the intuitive way to place a
finite streak — by a center point in the PIXEL frame plus an orientation and
length — and converts to the stored Hesse `rho`/`theta`/`s_center` via
`astro_lfd.geom.Line2D`. This is the *only* LSST touch point in `streak.py`: it
is lazy-imported inside that method so the module and `get_signal()` stay
stack-independent, preserving the numpy-only simulation path.

## Design Decisions

The radiometric simulation is always done in **electrons** because the noise
model requires it: sky and streak shot noise are Poisson (valid only on
electron counts) and read noise is quoted in electrons.  The final image and
variance are then converted to the requested output ``unit`` via a single
scalar calibration factor ``calib`` (electrons per output unit).

I/O is provided in two formats: compact ``.npz`` (no LSST dependency) for numpy
development, and ``lsst.afw.image.ExposureF`` FITS (LSST Science Pipelines
compatible).  Converters between the two are included.
