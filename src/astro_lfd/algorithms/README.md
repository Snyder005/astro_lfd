# algorithms

## Overview

Linear feature detection tasks. Each LFD implementation follows the standard
template of pre-processing, detection, and post-processing steps.

## Architecture

**LFD template.** Each LFD task implementation is an LSST `Task` that follows 
the standard template of separating the task into three sequential steps 
(`preprocess`, `detect`, and `postprocess`).

**HasTimings hierarchy.** `HasTimings` is a protocol that defines structural
 behavior of LFD taks for timing processing steps.

`KHTDetectTask` finds straight linear features (satellite streaks, and similar
signals) in an `lsst.afw.image.Exposure`.  It reproduces the detection stages
of `lsst.meas.algorithms.maskStreaks.MaskStreaksTask` -- Canny edge extraction,
`lsst.kht` line finding, recursive k-means clustering, and a Moffat 
line-profile fit -- but emits its results as a `~lsst.afw.table.SourceCatalog`
of canonical line segments (via `~astro_lfd.table.streakAdapter.StreakAdapter`)
instead of a mask plane.

`ADRTDetectTask` finds lines with the approximate discrete Radon transform
(`adrt` package). `detect` runs the forward ADRT on a square power-of-two
image and hands the `(4, 2N-1, N)` accumulator to `_find_peaks`, which returns
`list[Line2D]` in the PIXEL frame of the grid the ADRT ran on (undoing any
binning is the caller's job). `adrtButterfly.py` holds the closed-form
ADRT <-> Hesse coordinate maps and the experimental "butterfly" moment analysis
of the region around a peak.

## Design Decisions

The profile fitter (`Line`, `LineProfile`) is imported from ``maskStreaks``
rather than reimplemented, so any difference in the fit itself is shared
between the tasks.

### ADRT multi-peak detection (`ADRTDetectTask._find_peaks`)

**Why the accumulator is normalised by `sqrt(L)`.** An ADRT cell sums `L`
pixels (the digital line length, `L = adrt(ones)`), so under white noise of
unit variance its variance is `L`. `S = A / sqrt(L)` has unit variance in every
cell, which makes a single MAD-scaled threshold valid across short and long
lines and turns the accumulator value into a significance. Dividing by `L`
instead (a mean) would let 4-pixel corner lines dominate. The line-length
accumulator is cached per `N` on the task because it costs as much as the
forward transform (1.5 s at N=4096).

**Why a height-axis local-contrast test exists (Stage 2).** A streak does not
produce one peak: it casts a "butterfly" of ridges across the slope columns of
its quadrant (and into the neighbouring quadrants across the seams). Those
wings are broad along the height axis, whereas the true ridge is a few cells
wide, so a cell's excess over the median of an annulus 12-40 cells away along
`h` separates ridge cells from wing cells cheaply, before any geometry is
computed. On a bright single streak this cut 2513 local maxima to 191.

**Why the non-maximum suppression uses a physical wing envelope.** The
remaining wing maxima cannot be removed with a fixed exclusion radius: the wing
of a bright streak spans the entire quadrant and a genuinely faint streak may
lie inside it. The suppression therefore predicts how bright a wing may be. A
line at angle `d` to a streak of angular ridge half-width `dW` overlaps the
streak over `sin(dW)/sin(d)` of its length, so relative to the peak the wing
amplitude in `S` is `sin(dW)/sin(d) * sqrt(L_peak / L_candidate)`. Measured
wings reach 0.5-0.9 of this model; real crossing streaks are >= 0.95. A
candidate is suppressed only when it is inside the cone `|dh| <= |ds| +
cone_pad` **and** below `wing_margin * model * value + wing_noise * sigma`. The
additive `wing_noise` (4 sigma) is needed because the maximum over the many
cells of a wing crest overshoots the noise-free envelope; 3 sigma left a
spurious peak on a short bright streak at one noise seed. The ridge half-width
is measured per peak (`_ridge_half_width`) because it depends on streak length
and PSF.

**Why candidates are mapped into the accepted peak's quadrant.** The cone is a
property of the transform in `(h, s)` index space, but wings of a streak near a
seam continue into the adjacent quadrant, where the indices are unrelated.
`_hesse_to_adrt(..., quadrant=q)` extrapolates any line into the accepted
peak's frame, so the cone test is applied to candidates from all four
quadrants. Without this, near-seam and very bright streaks left 3-5 residual
peaks.

**Why the parabolic refinement is the default.** The butterfly moment analysis
(`extract_segment_adrt`) is exact on noiseless data but its second moments span
the full height axis of every slope column, so at unit per-pixel noise the
noise moment dominates and the recovered line is wrong by tens of pixels.
`refine="butterfly"` is kept as an opt-in for high-SNR or background-suppressed
accumulators; the three-point parabola along each axis gives ~0.5 px / 0.05 deg
at N=1024 and costs nothing.

**Accepted limitation.** A faint streak crossing a much brighter one inside the
cone with amplitude below the envelope is indistinguishable from a wing. At
N=1024 a 1 sigma/px streak next to a 10 sigma/px streak is suppressed; 1.5
sigma/px is recovered even next to 30 sigma/px.

**Arguments** (all keyword-only with defaults; deliberately not Task Config
fields so they can be swept without a config round-trip):

| Argument | Default | Role |
| --- | --- | --- |
| `line_length` | `None` | Precomputed `adrt(ones)`; cached per `N` when omitted |
| `k` | 6.0 | Detection threshold, MAD-sigma above the quadrant median of `S` |
| `k_contrast` | 5.0 | Height-axis local-contrast threshold, MAD-sigma |
| `footprint` | (7, 31) | Local-maximum footprint in `(h, s)` cells |
| `contrast_annulus` | (12, 40) | Height offsets whose median is the local background |
| `wing_margin` | 1.0 | Multiplier on the physical wing envelope (capped at 1) |
| `wing_noise` | 4.0 | Additive noise allowance on the envelope, MAD-sigma |
| `cone_pad` | 4 | Extra height cells on the wing cone |
| `min_length` | 64.0 | Minimum digital line length for a candidate |
| `max_peaks` | 50 | Cap on accepted peaks |
| `dedup_rho`, `dedup_theta` | 3.0 px, 0.5 deg | Seam de-duplication tolerance in Hesse space |
| `refine` | `"parabolic"` | `"none"`, `"parabolic"` or `"butterfly"` |
| `return_peaks` | `False` | Also return `AdrtPeak` descriptors (indices, value) |

**Preprocessing recommendations (not implemented here).** The detector assumes
the input has unit white noise and zero background: divide by the per-pixel
noise (`D / sqrt(V)`), subtract or fit out the sky, and interpolate or zero bad
pixels while passing the valid-pixel mask through `adrt` as `line_length`, so
masked lines are normalised by their valid pixel count. A matched filter along
the height axis (kernel ~ PSF FWHM) would raise the peak significance of
PSF-broadened streaks by roughly `sqrt(FWHM)`.

**Performance.** N=1024: ~0.5 s per image including the forward ADRT. N=4096:
forward ADRT 1.4 s, `_find_peaks` 6.7 s, dominated by the `maximum_filter`
and the per-quadrant statistics.
