# Closed-form line-segment moments from the Approximate Discrete Radon Transform

A methods report on the ADRT "butterfly" analysis as implemented in
`astro_lfd`. It is written to be adaptable into the methods section of a future
research project: it states the physical model, derives the moment law from it,
gives the closed-form inversion, and documents the algorithm and its numerical
validation. The tone is deliberately implementation-grounded — every equation
maps to a symbol in `src/astro_lfd/algorithms/adrtDetect.py`.

- **Companion documents.** Derivation summary and validation tables:
  [`butterfly.md`](butterfly.md). ADRT geometry and conventions:
  [`design.md`](design.md). Implementation status:
  [`butterfly-implementation-plan.md`](butterfly-implementation-plan.md).
- **Status:** derived, implemented, and numerically validated (2026-09).

---

## 1. Motivation and physical setting

Astronomical linear features — satellite streaks, meteor and airplane trails,
optical ghosts, cosmic-ray tracks — are elongated sources that a standard
point-source detection and deblending pipeline does not describe well. In the
LSST Camera survey, characterizing such features means recovering (i) their
orientation, (ii) their position, and (iii) their extent (length and transverse
width). Crucially, the same *image second moments* that catalogs use to
characterize ordinary sources — the adaptive moments `(XX, XY, YY)` — are a
natural and morphology-agnostic description of a streak: they encode orientation
and both spatial scales without assuming a particular light profile.

The Radon / Hough family of transforms is the classical tool for finding lines:
each maps image content into a parameter space in which a straight feature
becomes a localized concentration ("peak"). The **Approximate Discrete Radon
Transform** (ADRT; Brady 1998; Press 2006; implementation Otness & Rim 2023)
computes exact partial sums of pixel intensities along families of *digital*
lines in `O(N² log N)` for an `N×N` image, with no edge detection — it
integrates the streak body coherently, which is the decisive advantage for
low-surface-brightness features.

The question this report answers is: **given a detected peak in the ADRT
accumulator, can we recover the streak's image moments — and hence its full
geometry — in closed form, without returning to image space?** The answer is
yes, and the result is *better determined* than the analogous Hough-space
construction of Xu, Shin & Klette (2015), because in the ADRT the projection
coordinate is an *affine* rather than trigonometric function of the line slope.

---

## 2. The ADRT accumulator as a family of image projections

### 2.1 Digital lines and the accumulator

For an `N×N` image `I(x, y)` (with `N` a power of two), `adrt.adrt(I)` returns an
array of shape `(4, 2N−1, N)`. The first axis indexes four **quadrants** `q`
that partition the line-angle range `[−90°, +90°]` into four 45° bands; within a
quadrant the remaining axes index a **height** `h ∈ {0, …, 2N−2}` and a **slope**
`s ∈ {0, …, N−1}`. Each cell `A[q, h, s]` is the sum of image intensity along a
single digital line — the ADRT's discrete approximation to the continuous Radon
line integral.

The key structural fact is that a **fixed quadrant-and-slope column**
`A[q, :, s]` is a family of *parallel* digital lines of one common slope,
indexed by their offset `h`. That column is therefore the image **projected onto
the intercept coordinate** of that slope.

### 2.2 The intercept coordinate and its affine index map

Write a line of slope `s = dy/dx` in slope–intercept form. Its intercept is

```
b = y − s·x .                                                            (1)
```

For a fixed `(q, s)` column, the accumulator value at height index `h` is (to the
digital-line approximation) the integral of `I` along the line `b = const`, so
**the column, viewed as a function of `b`, is the intensity-weighted projection
of the image onto the `b`-axis.** The accumulator-value-weighted moments of a
column equal the image-intensity-weighted moments of the scalar field
`b = y − s·x`. This is the move that turns a statement about accumulator columns
into a statement about ordinary image moments.

Two closed-form maps connect accumulator indices to the physical `(s, b)`
(function `_slope_intercept_map`, verified to floating-point precision against
the full ADRT coordinate tables):

1. **Slope value** is the quadrant-selected map of the slope index — one of the
   four combinations
   ```
   s(q, s_idx) ∈ { +s_idx/(N−1),  −s_idx/(N−1),  +(N−1)/s_idx,  −(N−1)/s_idx } ,   (2)
   ```
   with `q` selecting the branch. No trigonometry is required.

2. **Intercept is exactly affine in the height index:**
   ```
   b = α(q, s_idx)·h + β(q, s_idx) ,                                     (3)
   ```
   with `α, β` closed-form in the same quadrant geometry (Appendix A). Because
   the map is affine, the raw integer-index moments transform to the physical
   intercept moments with no distortion of functional form (§3.2).

This affineness is exactly what makes the variance law below a *global*
quadratic; in the Hough transform the projection coordinate
`d = x·cos α + y·sin α` is trigonometric in the parameter `α`, and the analogous
law is only locally quadratic near the peak.

---

## 3. The moment law

### 3.1 Centroid and variance in terms of image moments

Let the image intensity `I(x, y)` have total flux `m00 = ∫I`, centroid
`(x_c, y_c) = (∫x I, ∫y I)/m00`, and central second moments

```
μ20 = ⟨(x−x_c)²⟩ ,   μ11 = ⟨(x−x_c)(y−y_c)⟩ ,   μ02 = ⟨(y−y_c)²⟩ ,        (4)
```

where `⟨·⟩` denotes the intensity-weighted average. These `(μ20, μ11, μ02)` are
the components of the 2-D inertia tensor of the light distribution and are
identical to the catalog adaptive moments `(XX, XY, YY)`.

For the fixed-slope column variable `b = y − s·x` of Eq. (1), linearity of the
expectation and variance gives, **exactly and for any intensity distribution**,

```
μ(s) ≡ ⟨b⟩   = y_c − s·x_c ,                          (linear in s)      (5)
V(s) ≡ Var(b) = μ02 − 2 μ11 s + μ20 s² .              (quadratic in s)    (6)
```

No thin-segment or small-angle approximation enters. The centroid trace `μ(s)`
is a straight line whose coefficients give the position; the variance trace
`V(s)` is a genuine parabola whose three coefficients **are** the inertia tensor.

### 3.2 From index moments to physical moments

In practice, for each column we compute the flux-weighted moments of the integer
height index `h` — the mean `⟨h⟩` and variance `Var(h)` — vectorized over the
whole slope band. The affine map Eq. (3) then carries them to the physical
intercept moments with no approximation:

```
μ(s) = α·⟨h⟩ + β ,        V(s) = α²·Var(h) .                             (7)
```

This is the computational heart of the method and the reason the whole slope
band reduces to a handful of array operations (no per-cell coordinate transform,
no Python loop over columns).

### 3.3 Uniform top-hat rectangle (the calibration model)

For a uniform rectangle of length `L` (longitudinal) and width `w` (transverse)
at line angle `φ0` (slope `s0 = tan φ0`), the body-frame second moments are
`Var(u) = L²/12`, `Var(v) = w²/12`, `Cov(u,v) = 0`. Rotating this diagonal tensor
into image axes,

```
μ20 = (L²/12) cos²φ0 + (w²/12) sin²φ0 ,
μ02 = (L²/12) sin²φ0 + (w²/12) cos²φ0 ,                                   (8)
μ11 = (L²/12 − w²/12) sin φ0 cos φ0 .
```

Substituting into Eq. (6) reproduces the intuitive form
`V(s) = [L²(s−s0)² + w²(1 + s s0)²] / [12(1+s0²)]`. The top-hat is the natural
validation source: its geometry is exactly known, so recovered `(L, w, φ0)` can
be compared to truth, and any residual is a *discretization* systematic (§5).

---

## 4. The closed-form inversion

Fit the two low-order traces over a band of slope columns around the peak:

```
V(s) = A s² + B s + C  (weighted quadratic),   μ(s) = β0 + β1 s  (weighted line).
```

Identifying with Eqs. (5)–(6),

```
(μ20, μ11, μ02) = (A, −B/2, C) ,    (x_c, y_c) = (−β1, β0) .             (9)
```

The `(μ20, μ11, μ02)` and the center are the **primary data products** — a
morphology-agnostic description valid for any streak, stored in `ADRTSegment`.
Orientation follows from the standard principal-axis formula,

```
φ0 = ½ atan2(2 μ11, μ20 − μ02) ,   θ = φ0 + π/2  (line-normal angle) ,   (10)
```

and `ρ = x_c cos θ + y_c sin θ` gives the Hesse-normal offset — a sub-pixel
refinement of the integer-peak orientation.

For the *top-hat interpretation* (simulations, and any application that assumes a
uniform-rectangle model), the eigenvalues of the inertia tensor are the
principal moments `L²/12` (major) and `w²/12` (minor), giving

```
L² = 6 [ (A+C) + √((A−C)² + B²) ] ,   w² = 6 [ (A+C) − √((A−C)² + B²) ] . (11)
```

This is implemented as `ADRTSegment.segment_dimensions()` → `_invert_inertia`,
kept deliberately separate from the moment extraction because `(L, w)` are a
model-dependent reading of the model-independent moments.

**Why this beats the Hough construction.** In Xu–Shin–Klette the variance law is
quadratic only near the vertex, the vertex is displaced from the true angle at
finite width, and recovering the aspect ratio requires the orientation to be
supplied externally. Here `V(s)` is a global quadratic whose three coefficients
over-determine `(φ0, L, w)` on their own; the slope is a *fitted output*, not an
input.

---

## 5. Numerical validation and systematics

Clean top-hat segments simulated in a 4096×4096 frame, transformed, and inverted
(`devel/butterfly_closed_form.py`; regression tests in
`tests/test_adrt_butterfly.py`) recover:

| quantity | performance (sweep: φ0∈{10–40}°, L∈{1000–2500}, w∈{4,8,16}) |
| --- | --- |
| length `L`   | mean +0.06 %, max \|·\| 0.68 % |
| angle `φ0`   | mean −0.0007°, max \|·\| 0.006° |
| center       | exact to ≪1 px |
| width `w`    | small positive bias (below) |

Two structural checks confirm the theory rather than a lucky fit: the recovered
`L`/`w` are **stable as the fit half-band grows** from ~40 to ~300 columns
(the law is globally quadratic, not a local parabolic approximation), and the
**slope is recovered without being supplied**.

**Width systematic.** The width carries a small *additive* bias in `w²`
(measured `w_hat² − w² ≈ few px²`, approximately independent of `w`) — a
digital-line analogue of Sheppard's discretization correction, dominant only at
sub-pixel widths. Because it is additive and roughly constant it is calibratable
and subtractable via the `width_bias` argument to `segment_dimensions`. A
Gaussian PSF adds a further known `σ_psf²` to each principal moment
(`Var → Var + σ_psf²`), separable from the geometry and deconvolvable when the
PSF is known. Characterizing both terms versus angle, quadrant, and blur is the
next calibration task.

**Domain of validity.** The moment law assumes a single, isolated uniform
feature occupying the column support. On a raw noisy full-frame exposure the sky
integrated along ~4096-pixel lines dominates the column moments; the method
therefore presumes a *conditioned* accumulator (sky-subtracted / significance
input, matched-filter and ridge windowing) and a reasonable peak. Robust
accumulator conditioning and multi-peak detection are prerequisites for real
data and are tracked separately. A per-column background subtraction (median,
clipped at zero) is applied as a first-order mitigation.

---

## 6. Algorithm summary

Given a detected peak `(q, h, s_idx)` and domain size `N`
(`extract_segment_adrt`):

1. **Band.** Take slope columns `s_idx ± half_band`, clipped strictly interior to
   `[1, N−1)` to avoid the quadrant seams.
2. **Map.** `_slope_intercept_map(q, s_cols, N)` → per-column `(slope, α, β)`
   (Eqs. 2–3), fully vectorized.
3. **Condition.** Subtract a per-column background (median, clipped ≥ 0).
4. **Moments.** Flux-weighted `⟨h⟩`, `Var(h)` per column; map to
   `μ(s) = α⟨h⟩+β`, `V(s) = α²Var(h)` (Eq. 7).
5. **Fit.** Flux-weighted quadratic `V(s)` and linear `μ(s)`.
6. **Extract.** `(μ20, μ11, μ02)`, center (Eq. 9), and refined `(ρ, θ)`
   (Eq. 10); package as `ADRTSegment`.
7. **(Optional) top-hat dimensions.** `segment_dimensions()` inverts the tensor
   to `(L, w, φ0)` (Eq. 11), used by the finite-segment representation in
   post-processing and by simulation validation.

Binning is handled outside the transform as an isotropic rescale of the pixel
grid: linear quantities (`ρ`, center) scale by `bin_size`, the second moments (in
px²) by `bin_size²`, and `θ` is unchanged.

---

## Appendix A — the affine intercept coefficients

With `n = s_idx/(N−1)`, `t = arctan(n)`, `cs = cos t + sin t`, `c = (N−1)/2`,
`k = (2N−1)/(2N)`, and the quadrant sign `s_q = +1` for `q ∈ {0,2}` else `−1`,
the line-normal angle is `θ = π/2 − angle(q)` with
`angle ∈ {t−π/2, −t, t, π/2−t}` for `q ∈ {0,1,2,3}`. Then

```
P = s_q · cs / (1 + n) ,
Q = c·(cos θ + sin θ) − s_q · N · cs · (k − ½) ,
α = P / sin θ ,     β = Q / sin θ ,
slope = −cos θ / sin θ   (equivalently the Eq. 2 combination).
```

These reproduce the former per-cell ADRT→Hesse transform exactly (checked to
~1e-10 px over all quadrants and slope indices), but evaluate the whole slope
band at once, eliminating the per-row transform and per-column loop.

---

## References

- M. L. Brady, "A fast discrete approximation algorithm for the Radon
  transform," *SIAM J. Comput.* 27(1), 1998.
- W. H. Press, "Discrete Radon transform has an exact, fast inverse and
  generalizes to operations other than sums along lines," *PNAS* 103(51), 2006.
- K. Otness, W. Rim, "adrt: approximate discrete Radon transform for Python,"
  *JOSS* 8(83), 2023.
- Z. Xu, B.-S. Shin, R. Klette, "Closed form line-segment extraction using the
  Hough transform," *Pattern Recognition* 48(12), 2015.
- Image second-moment / principal-axis (inertia tensor) orientation and
  axis-length formulas — standard (cf. Hu moments; adaptive-moment source
  characterization).
