# LFD_DESIGN.md — Linear Feature Detection framework

Unified design report for **Linear Feature Detection (LFD)** in astronomical
difference images. This document describes the **detector-agnostic framework**:
the shared task shape, inputs/outputs, coordinate convention, and
astronomy-specific modifications that *every* detector inherits. Method-specific
design lives per detector under [`docs/detectors/<method>/`](detectors/).

- **Status:** living design doc. Evolving — revise as experiments settle.
- **Scope:** detection of linear features (satellite/aircraft streaks, some
  cosmic-ray tracks, diffraction spikes) as line parameterizations. Notebooks in
  this repo are scratch and are **not** treated as a spec.
- **Detectors:** KHT is the validated reference
  ([`knowledge/kht-detect.md`](../knowledge/kht-detect.md)); ADRT is the first
  non-Hough peer ([`docs/detectors/adrt/design.md`](detectors/adrt/design.md)).
  Further methods (Line Segment Detector, Frangi vesselness, YOLO) fit the same
  interface.

---

## 1. The shared shape

Every LFD detector maps image content into a `(ρ, θ)` line-parameter space and
locates concentrations there, following the same five-stage shape:

**prepare → transform → detect peaks → post-process → emit Hesse-form lines.**

Detectors differ **only in the line-finding core** (the transform + peak/line
extraction). Everything around it — the input contract, the fit-weights rule,
the output `SourceCatalog`, and the coordinate handling — is shared. The
reusable task template that encodes this is documented in
[`knowledge/detector-task.md`](../knowledge/detector-task.md); a new method is
added by swapping the core and reusing the rest.

### Reference core (KHT)

The validated reference detector (`astro_lfd.meas.detectStreaks.KHTDetectTask`,
ported from the MixCOATL prototype and validated against
`lsst.meas.algorithms.maskStreaks.MaskStreaksTask`):

1. Threshold / `DETECTED` plane → seed pixels.
2. Canny edge transform.
3. Mask bad pixels.
4. Kernel Hough Transform (`lsst.kht.find_lines`) + clustering.
5. Cluster pairs of edges of a feature; per cluster fit a profile and emit a line.

**Output:** lines in **Hesse normal form** `ρ = x·cosθ + y·sinθ`.

Alternative cores plug into stages 2–5 while keeping the same inputs and output
— e.g. the ADRT core drops Canny and transforms intensities directly (see its
design doc).

---

## 2. Inputs and preconditions

All detectors consume the same inputs:

- **Difference image** `D` — float, sky/background ≈ 0.
- **Bad-pixel mask** `M` — `True` = bad/ignore.
- **Variance plane** `V` — per-pixel variance (> 0 where valid).

Real inputs are difference images from the Butler; synthetic inputs come from
[`astro_lfd.utils.testdata`](../knowledge/testdata.md). A `DETECTED` mask plane
seeds detection (see the detector-task input contract).

Individual detectors may impose **additional** preconditions on the array they
transform (e.g. the ADRT requires a power-of-two square, so its front-end pads).
Those are documented in the per-detector design docs, not here.

---

## 3. Output format

A `SourceCatalog` (one record per accepted line), written through
`StreakAdapter` — geometry (`line_rho`, `line_theta`, `line_u_center`,
`line_length`, `line_center_{x,y}`) in the absolute pixel frame with
`theta ∈ [0, π)`, profile-fit quality fields, a per-line `Footprint`, and sky
`coord` when a WCS is present. See
[`knowledge/detector-task.md`](../knowledge/detector-task.md) and
[`knowledge/geom-line.md`](../knowledge/geom-line.md).

---

## 4. Coordinate handling (shared)

The profile fit works in an **image-array frame centered on the image**
(`arange(n) − n/2`, theta in **degrees**), mapped to absolute pixels via
`Line2D(rho, theta·deg).translated(Extent2D(bbox.getCenter()))` then
`.intersection(bbox)`, using `exposure.getBBox()`. `Line2D` canonicalizes theta
to `[0, π)`. **Any detector whose core emits lines in a different frame** (e.g.
the ADRT `(offset, angle)` space) must convert into this same image-centered
`(rho, theta[deg])` convention before the shared fit + output steps.

---

## 5. Astronomy-specific modifications (common to all detectors)

1. **Variance-aware detection** — use `D/√V` (or variance weighting) so
   detection significance is calibrated across the frame.
2. **Mask handling** — bad pixels must be neutralized before/within the core so
   they neither create nor suppress spurious lines; the exact mechanism is
   core-specific (edge-map masking for KHT; transform-the-mask for ADRT).
3. **Detector-frame bookkeeping** — padding/tiling and coordinate mapping back to
   the sky/pixel frame for non-square, non-power-of-two detector data.
4. **PSF-matched conditioning** — model the streak cross-section (~PSF FWHM) at
   the appropriate stage (KHT's Gaussian vote kernel; ADRT's offset matched
   filter).
5. **Physical vetoes** — reject detector-aligned artifacts, bleed trails, and
   cosmic rays; optional WCS conversion of `(ρ,θ)` to on-sky orientation.
6. **False-positive control** — critical at survey scale, where even low
   false-positive rates flood downstream catalogs (e.g. ADRT back-projection
   verification).

---

## 6. Per-detector design docs

- **KHT** (reference): [`knowledge/kht-detect.md`](../knowledge/kht-detect.md)
  and the validation notes under
  [`docs/detectors/kht/`](detectors/kht/).
- **ADRT**: [`docs/detectors/adrt/design.md`](detectors/adrt/design.md);
  verified backend API in [`knowledge/adrt-api.md`](../knowledge/adrt-api.md).
- *Future methods:* add `docs/detectors/<method>/` + a
  `knowledge/<method>-*.md` note.

---

## References

- Kernel-based Hough Transform: Fernandes & Oliveira, *"Real-time line detection
  through an improved Hough transform voting scheme,"* Pattern Recognition, 2008
  (clusters approximately-collinear edge pixels; votes with an oriented
  elliptical-Gaussian kernel for a cleaner accumulator).
- Approximate Discrete Radon Transform: Otness & Rim, JOSS 2023 (see the ADRT
  design doc and `knowledge/adrt-api.md`).
- Hough transform / Hesse normal form `ρ = x·cosθ + y·sinθ`.
