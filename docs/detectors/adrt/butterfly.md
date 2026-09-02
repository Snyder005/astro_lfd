# ADRT butterfly analysis — closed-form line-segment geometry

Closed-form extraction of line-segment geometry (**slope, position, length,
width**) directly from the ADRT slope–height accumulator around a detected peak,
without returning to image pixel space. This is the ADRT analogue of the
Hough-space "butterfly" moment analysis of Xu, Shin & Klette (2015), and it
turns out to be **cleaner and more completely determined** than the Hough case.

- **Status:** derived and numerically validated (2026-08-22). Continuous-slope
  moment law confirmed exact; inversion recovers geometry to sub-percent
  (length, angle) with a small, subtractable width bias.
- **Grounding:** conventions and the ADRT geometry in [`design.md`](design.md);
  the closed-form slope/intercept map (`_slope_intercept_map`), the moment
  extractor (`extract_segment_adrt` → `ADRTSegment`), and the analytic peak
  placement (`_hesse_to_adrt`) in `src/astro_lfd/algorithms/adrtDetect.py`; and
  the pixel/Hesse conventions in the `devel/` coordinate notes. Validation
  script: `devel/butterfly_closed_form.py` (scratch, gitignored).

---

## 0. TL;DR — the result

Around a peak, treat each ADRT slope column as a probability distribution over
the (continuous) intercept coordinate and measure its intensity-weighted
**centroid** `μ` and **variance** `V` as functions of the continuous line slope
`s`. Then, for a uniform top-hat segment, **exactly**:

```
μ(s) = y_c − s · x_c                         (linear in s)     → position
V(s) = μ20 · s² − 2 μ11 · s + μ02            (quadratic in s)  → orientation, L, w
```

where `(μ20, μ11, μ02)` are the *central second moments of the image intensity*
(the 2-D inertia tensor). The three fitted quadratic coefficients
`(A, B, C) = (μ20, −2μ11, μ02)` invert in closed form to the full segment
geometry:

```
tan(2 φ0) = −B / (A − C)                     line angle (dy/dx slope s0 = tan φ0)
L² = 6 [ (A + C) + √((A−C)² + B²) ]          longitudinal length²
w² = 6 [ (A + C) − √((A−C)² + B²) ]          transverse width²
(x_c, y_c) = (−β1, β0)   from μ(s)=β0+β1 s    center in the ADRT pixel grid
```

Unlike the Hough derivation — which needs `s0` known to turn the vertex
displacement into an aspect ratio — the ADRT variance law is a *genuine*
quadratic in `s` whose three coefficients over-determine `(φ0, L, w)` on their
own. `s0` falls out of the fit.

---

## 1. Setup and the key identity

### The ADRT column is a projection onto the intercept axis

For a fixed quadrant `q` and slope index, the ADRT accumulates image intensity
along a family of parallel digital lines of a single slope. Cell value
`A[q, h, s_idx]` is (to the ADRT's digital-line approximation) the line integral
of the image along the line of slope `s` whose intercept is indexed by `h`. So a
**column** `A[q, :, s_idx]` is the image projected onto the intercept
coordinate

```
b = y − s x            (y-intercept of the line of slope s through (x, y))
```

The accumulator-value-weighted moments of a column are therefore identical to
the *image-intensity-weighted* moments of the scalar field `b = y − s x`
evaluated over the image. This is the crucial move — it converts a statement
about accumulator columns into a statement about ordinary image moments, which
we can compute in closed form for a known shape.

> **Note on the height→intercept mapping.** The derivation uses the plain
> slope–intercept intercept `b = y − s x`. The ADRT height index `h` is an
> *exact affine* reparameterization of `b` (`design.md`): for a fixed
> quadrant/slope column, `b = α·h + β`, and the continuous slope `s` is the
> quadrant-selected map of the slope index (the four combinations of
> `±s/(N−1)`, `±(N−1)/s`). Both `(s, α, β)` are closed form in `(q, s_idx, N)` —
> see `_slope_intercept_map`. A per-column affine rescale of the axis does not
> change which curve (linear/quadratic) the moments follow, only its
> coefficients: the raw integer-index moments `⟨h⟩`, `Var(h)` map to the
> physical `μ(s) = α⟨h⟩ + β` and `V(s) = α²·Var(h)` with no approximation. This
> is why the whole slope band is a handful of vectorized array ops — no per-cell
> coordinate transform and no Python loop.

### Centroid and variance in terms of image moments

Let the image intensity be `I(x, y)` with total mass `m00 = ∫I`, centroid
`(x_c, y_c) = (∫xI, ∫yI)/m00`, and central second moments

```
μ20 = ⟨(x−x_c)²⟩,   μ11 = ⟨(x−x_c)(y−y_c)⟩,   μ02 = ⟨(y−y_c)²⟩
```

(angle brackets = intensity-weighted average). For a fixed slope `s`, the column
variable is `b = y − s x`, so:

```
μ(s) = ⟨b⟩ = y_c − s x_c
V(s) = Var(b) = Var(y − s x)
     = Var(y) − 2 s Cov(x, y) + s² Var(x)
     = μ02 − 2 μ11 s + μ20 s²
```

Both are **exact** for any intensity distribution — no thin-segment or
small-angle approximation. `μ(s)` is exactly linear; `V(s)` is exactly
quadratic. The coefficients are the components of the image's second-moment
(inertia) tensor.

This is why, in the original `butterfly_investigation.py`, the centroid-vs-column
was linear and the variance-vs-column was quadratic: those are the two lowest
image moments viewed through the ADRT's slope parameterization. What the
brainstorming missed is that the quadratic coefficients *are the inertia tensor*,
which is what makes the inversion fully determined.

---

## 2. Moments of a uniform top-hat rectangle

Model the streak as a uniform rectangle of length `L` (longitudinal) and width
`w` (transverse), centered at `(x_c, y_c)`, with its long axis at line angle
`φ0` to the x-axis (slope `s0 = tan φ0`). In body coordinates `(u, v)` (along,
across):

```
Var(u) = L²/12,   Var(v) = w²/12,   Cov(u, v) = 0
```

Rotating the diagonal body tensor `diag(L²/12, w²/12)` by `φ0` into image axes:

```
μ20 = (L²/12) cos²φ0 + (w²/12) sin²φ0
μ02 = (L²/12) sin²φ0 + (w²/12) cos²φ0
μ11 = (L²/12 − w²/12) sinφ0 cosφ0
```

Substituting into `V(s) = μ20 s² − 2μ11 s + μ02` and using `s0 = tan φ0`
reproduces the brainstorm's

```
V(s) = [ L² (s − s0)² + w² (1 + s s0)² ] / [ 12 (1 + s0²) ]
```

after algebra — the two forms are identical. The inertia-tensor form is the one
to fit, because its coefficients invert directly.

---

## 3. The inversion (three coefficients → three unknowns)

Given a fit `V(s) = A s² + B s + C`, identify `A = μ20`, `B = −2μ11`, `C = μ02`.
The eigenvalues of the 2-D inertia tensor `[[μ20, μ11],[μ11, μ02]]` are

```
λ± = (A + C)/2 ± ½ √((A − C)² + B²)
```

The principal moments of a uniform rectangle are `L²/12` (major) and `w²/12`
(minor), so

```
L² = 12 λ+ = 6 [ (A + C) + √((A − C)² + B²) ]
w² = 12 λ− = 6 [ (A + C) − √((A − C)² + B²) ]
```

The major-axis orientation (line angle) is the standard image-moment formula

```
tan(2 φ0) = 2 μ11 / (μ20 − μ02) = −B / (A − C)      →   φ0 = ½ atan2(−B, A − C)
```

and the slope is `s0 = tan φ0`. Position comes from the linear centroid fit
`μ(s) = β0 + β1 s`:

```
x_c = −β1,   y_c = β0
```

All four geometric quantities `(φ0, L, w, x_c, y_c)` are recovered from the two
low-order moment fits. Endpoints follow from `(x_c, y_c)`, `φ0`, and `L`.

---

## 4. Numerical validation

`devel/butterfly_closed_form.py` simulates clean top-hat segments in a
4096×4096 frame, measures `μ(s)` and `V(s)` in **continuous** ADRT coordinates
(each column's raw-index moments mapped through the closed-form affine
`b = α·h + β`), fits, inverts, and compares to truth.

**Single case** (`φ0 = 25°, L = 1500, w = 8, center = (2048, 2048)`):

| quantity | truth | recovered | error |
| -------- | ----- | --------- | ----- |
| slope s0 | +0.46631 | +0.46630 | exact to 5 digits |
| angle φ0 | 25.000° | 25.000° | < 0.001° |
| length L | 1500.0 | 1499.0 | −0.07 % |
| width w  | 8.0 | 8.18 | +2.2 % |
| center   | (2048, 2048) | (2048.00, 2048.00) | exact |

**Sweep** (48 cases: `φ0 ∈ {10,20,30,40}°`, `L ∈ {1000,1500,2000,2500}`,
`w ∈ {4,8,16}`):

| quantity | mean error | std | max |abs| |
| -------- | ---------- | --- | -------- |
| length L | +0.06 % | 0.19 % | 0.68 % |
| width w  | +3.27 % | 3.27 % | 10.9 % |
| angle φ0 | −0.0007° | 0.003° | 0.006° |

**Length and angle are essentially exact.** The width error is a **constant
additive bias in `w²`** (Sheppard-type discretization correction), not a scale
error: measured `w_hat² − w² ≈ 2.6 px²` independent of `w` (checked at
`w = 4, 8, 16, 32`), and order 1–3 px² across angles. It dominates the relative
error only at small `w`. It can be calibrated out by subtracting a fixed
`Δ(w²) ≈ few px²` (a digital-line analogue of Sheppard's `1/12` correction).

Two further checks:

- **Global quadraticity.** `V(s)` is quadratic over the *whole* slope axis, not
  just near the vertex: recovered `L`/`w` are stable as the fit half-band grows
  from 40 to 300 columns (length error stays < 1 %). This matches the exact-in-`s`
  theory — the fit is not relying on a local parabolic approximation.
- **Slope is a free parameter.** No case supplies `s0`; it is recovered from the
  fit to < 0.01°, confirming the ADRT inversion is fully determined (the Hough
  version needs `α0` externally).

---

## 5. Why the ADRT case is better than Hough

| | Hough (Xu–Shin–Klette) | ADRT (this work) |
| --- | --- | --- |
| Projection coordinate vs. parameter | `d = x cosα + y sinα` — **trigonometric** in `α` | `b = y − s x` — **affine** in `s` |
| Variance law | `σ²(α) = (L² sin²Δα + T² cos²Δα)/12`, quadratic only near the vertex | `V(s) = μ20 s² − 2μ11 s + μ02`, **globally exact quadratic** |
| Vertex | displaced from `α0` at finite width; needs `α0` known to extract aspect ratio | fit coefficients *are* the inertia tensor; `(φ0, L, w)` all determined |
| Position | linear mean curve | `μ(s) = y_c − s x_c`, exact |

The affine (slope–intercept) parameterization is what makes `V(s)` a true
quadratic in the fit variable and the inversion closed-form and complete. This is
a genuinely useful discovery for a future ADRT-detector paper.

---

## 6. Caveats and open items

1. **Continuous vs. index coordinates.** The exact quadratic holds in continuous
   `(s, b)`. The raw integer ADRT column index `s_idx` maps to `s` through a
   *nonuniform* quadrant map (`design.md`); fitting against `s_idx` directly is
   only locally quadratic. Always convert to the continuous slope and intercept
   via `_slope_intercept_map` first (as `extract_segment_adrt` and the
   validation do).
2. **Width bias.** The additive `w²` offset from digital-line discretization
   should be characterized (dependence on angle, quadrant, PSF blur) and
   subtracted. Sub-pixel widths (`w ≲ 4`) are the least accurate.
3. **Isolated-segment assumption.** The moment law is derived for a single
   uniform rectangle occupying the column support. Real accumulators include
   background, neighbouring features, and the ADRT's cross-quadrant leakage. A
   local background subtraction / windowing in the accumulator (matched to the
   peak) is needed before the moment sums — see the implementation plan.
4. **PSF / blurred edges.** A Gaussian-blurred top-hat adds `σ_psf²` to each
   principal moment (`Var → Var + σ_psf²`), i.e. another additive term in
   `(L², w²)`. This is separable from the geometry and can be deconvolved if the
   PSF width is known.
5. **Quadrant boundaries.** Near `s_idx = 0` or `N−1` (angles 0/45/90/135°) the
   peak column can straddle a quadrant seam; the moment window must stay within
   one quadrant or use `stitch_adrt` with seam care.

---

## References

- Z. Xu, B.-S. Shin, R. Klette, "Closed form line-segment extraction using the
  Hough transform," *Pattern Recognition* 48(12), 2015 — the Hough-space
  precedent (variance/centroid moment analysis).
- ADRT geometry and the closed-form slope/intercept map: [`design.md`](design.md)
  and `src/astro_lfd/algorithms/adrtDetect.py` (`_slope_intercept_map`,
  `_hesse_to_adrt`).
- Image second-moment (inertia tensor) orientation/axis-length formulas — standard
  (e.g. Hu moments; principal-axis analysis).
