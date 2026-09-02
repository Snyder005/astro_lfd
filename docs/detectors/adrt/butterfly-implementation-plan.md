# ADRT butterfly analysis — implementation plan

Plan to turn the validated closed-form result in [`butterfly.md`](butterfly.md)
into a reusable analysis that estimates line-segment geometry
(**slope/height → ρ,θ; length; width; endpoints**) directly from the ADRT
accumulator around a detected peak. This is the ADRT counterpart of the
Xu–Shin–Klette Hough butterfly, and it slots into the existing `ADRTDetectTask`
between peak finding and post-processing.

- **Status:** steps 1–3 implemented (2026-08-22); extractor simplified and
  moments-first (2026-09-02). Width-bias/PSF calibration (step 6) and robust
  accumulator conditioning remain.
- **Depends on:** the closed-form derivation and inversion in
  [`butterfly.md`](butterfly.md) (validated in `devel/butterfly_closed_form.py`).

## Implementation status

- **Done:** `ADRTSegment` (moments-first result), `extract_segment_adrt`,
  `_slope_intercept_map`, `_invert_inertia` in
  `src/astro_lfd/algorithms/adrtDetect.py`; `_find_peaks` returns integer
  `(q, h, s)` peaks; `detect` runs the extractor and applies `bin_size` scaling
  via `_apply_bin_size`; `postprocess` builds a `LineSegment2D` from
  center/angle and the top-hat length from `ADRTSegment.segment_dimensions()`,
  and persists `line_width` (added to the schema in `streakAdapter.py`). Tests in
  `tests/test_adrt_butterfly.py` (53 cases): pure inversion is exact; on clean
  signal the recovered `(mu20, mu11, mu02)` match the known rectangle tensor,
  length < 1 %, angle < 0.05°, width within the additive bias.
- **Simplification (2026-09-02).** The per-column Python loop and the per-cell
  `_adrt_to_hesse` transform were removed. The slope value is the trig-free
  4-combo map of the slope index, and the height→intercept map is exact affine
  (`b = α·h + β`); both are closed form in `(q, s_idx, N)` via
  `_slope_intercept_map`, so the moment sums are fully vectorized over the slope
  band (verified to reproduce the prior coefficients and center to
  floating-point precision). `_adrt_to_hesse` was deleted; `_hesse_to_adrt` is
  kept for analytic peak placement in the sim tests. The `A/B/C`, `slope`, and
  `length`/`width` fields were dropped from the result: `(A, B, C) =
  (mu20, −2mu11, mu02)` and the top-hat length/width are a derived
  interpretation (`segment_dimensions`), whereas the 2-D central moments (=
  catalog `XX, XY, YY`) are the primary product for arbitrary streak
  morphologies. **Quadrant-agnostic reflection scheme: not implemented** — the
  direct per-quadrant affine map is already trivial and exact, so reflecting/
  transposing to a canonical quadrant would add indirection with no gain.
- **Known gap (deferred):** on a *noisy full-frame* exposure the estimate is
  dominated by background integrated along the 4096-pixel lines (the
  isolated-segment caveat, [`butterfly.md`](butterfly.md) §6.3). The estimator is
  correct given a conditioned accumulator; robust accumulator conditioning
  (sky-subtracted/significance input per `design.md` Steps 0–1, matched-filter +
  ridge windowing per Step 4) and multi-peak detection (issue #7) are the
  prerequisites for real data and are tracked separately.

---

## 1. Where it fits in the existing task

`ADRTDetectTask` (`src/astro_lfd/algorithms/adrtDetect.py`) currently does:

```
run → preprocess → detect(_find_peaks → extract_segment_adrt) → postprocess
```

`_find_peaks` returns the global-argmax peak `(q, h, s_idx)`, `detect` runs the
butterfly extractor per peak, and `postprocess` builds a finite `LineSegment2D`
from the recovered center/angle/length (the top-hat `segment_dimensions`).

The butterfly analysis consumes a peak `(q, h, s_idx)` plus a local accumulator
window and returns the segment moments. It uses the closed-form
`_slope_intercept_map` for the continuous-coordinate mapping, so it introduces no
new coordinate math — only vectorized moment sums and the closed-form inversion.

---

## 2. Proposed API

A single pure function plus a small result container, kept independent of the
LSST task machinery (like the existing transform helpers) so it is unit-testable
without the stack:

```python
@dataclass
class ADRTSegment:
    rho: float            # Hesse ρ (ADRT pixel grid), from the fitted moments
    theta: float          # Hesse θ (rad), line-normal angle
    center_x: float       # px (ADRT pixel grid), first moments
    center_y: float
    mu20: float           # 2-D central second moments (px²) = catalog XX/XY/YY
    mu11: float
    mu02: float
    var_residual: float   # goodness of the variance-quadratic fit
    n_columns: int

    def segment_dimensions(self, width_bias: float = 0.0) -> tuple[float, float, float]:
        """(length, width, phi0) top-hat interpretation via _invert_inertia."""
        ...

def extract_segment_adrt(
    adrt_result: NDArray,     # (4, 2N-1, N)
    q: int, h: float, s_idx: int,   # peak (from _find_peaks)
    N: int,
    *,
    half_band: int = 90,           # slope columns each side of the peak
    background: str = "median",    # per-column background subtraction
) -> ADRTSegment:
    ...
```

Placement: lives in `src/astro_lfd/algorithms/adrtDetect.py` next to
`_slope_intercept_map` / `_hesse_to_adrt`, keeping the ADRT math in one place
(split to a sibling `adrtButterfly.py` only if it grows).

---

## 3. Algorithm (per detected peak)

1. **Select the slope band.** Columns `s_idx − half_band … s_idx + half_band`,
   clipped to `[1, N−1]` and to the peak's quadrant (do not cross seams).
2. **Map to continuous coordinates.** For the whole band at once,
   `_slope_intercept_map(q, s_cols, N)` → `(slopes, α, β)`: the slope value
   (quadrant 4-combo map) and the exact affine intercept map `b = α·h + β`.
3. **Condition each column.** Subtract a local background (median of the column,
   or a windowed estimate around the ridge) so the moment sums see the segment,
   not the sky pedestal. Optionally restrict to a height window around the ridge
   centroid to limit contamination.
4. **Column moments.** Vectorized weighted moments of the integer height index
   per column, mapped to continuous `b`: `μ(s) = α⟨h⟩ + β`, `V(s) = α²·Var(h)`.
5. **Fit.** Weighted least squares: linear `μ(s) = β0 + β1 s`, quadratic
   `V(s) = A s² + B s + C`. Weight by column total flux (SNR) so the peak column
   dominates and far, contaminated columns matter less.
6. **Identify moments.** `(A, B, C) = (μ20, −2μ11, μ02)` and
   `(x_c, y_c) = (−β1, β0)` — the primary product. Orientation is
   `φ0 = ½ atan2(2μ11, μ20−μ02)`, `θ = φ0 + 90°`. The top-hat length/width are a
   derived interpretation via `ADRTSegment.segment_dimensions(width_bias)` →
   `_invert_inertia` (closed form, `butterfly.md` §3).
7. **Refine ρ, θ.** θ from the fitted moment tensor (normal angle = φ0 + 90°) and
   ρ from the centroid line through `(x_c, y_c)` — a sub-pixel refinement of the
   integer peak.
8. **Package** into `ADRTSegment` with fit diagnostics.

`postprocess` then builds a `LineSegment2D` from `(center, φ0, L)` via
`LineSegment2D.from_center_length` (already exists) instead of clipping an
infinite line — giving true endpoints.

---

## 4. New / changed source

- `src/astro_lfd/algorithms/adrtDetect.py`
  - `ADRTSegment`, `extract_segment_adrt`, and small private helpers
    `_slope_intercept_map`, `_invert_inertia`.
  - `_find_peaks` returns the peak indices `(q, h, s_idx)` so the extractor can
    run.
  - `detect` calls `extract_segment_adrt` per peak, applies `bin_size` scaling
    (`rho`/`center` by `bin_size`, second moments by `bin_size²`; θ unchanged),
    and passes segments to `postprocess`.
  - `postprocess` consumes segments (center, angle, and `segment_dimensions`
    length/width) and writes endpoints + width to the streak catalog.
- `src/astro_lfd/table/streakAdapter.py` — `line_width` field added to the
  schema; `setLineSegment` carries the endpoints via ρ/θ/s_center/length.

## 5. Tests (`tests/`)

- **Unit (no stack):** `_invert_inertia` inverts a known `(μ20, μ11, μ02)` tensor
  to `(length, width, phi0)` exactly.
- **Integration (sim, needs stack):** parametrized over a `(φ0, L, w)` grid;
  assert the recovered `(μ20, μ11, μ02)` match the known rectangle tensor,
  `|ΔL|/L < 1 %`, `|Δφ0| < 0.05°`, and `|Δw|` within the calibrated bias
  tolerance (`< 15 %` at `w ≥ 4`, tighter after bias subtraction).
- **Regression:** width-bias constant `w_hat² − w²` stable vs. `w` (guards the
  Sheppard-correction assumption).

## 6. Calibration work (before trusting width)

- Characterize the additive `w²` bias vs. angle, quadrant, and PSF blur; store a
  small calibration (constant or low-order in angle) and pass it as the
  `width_bias` argument to `segment_dimensions`.
- Extend the derivation for a **PSF-blurred** top-hat: `Var → Var + σ_psf²` adds
  a known term to both principal moments; deconvolve when the PSF is known.

## 7. Sequencing

1. Land the pure estimator + unit tests (no stack) — smallest, self-contained.
2. Add the sim integration test (reuses the validated devel script).
3. Wire into `detect`/`postprocess` + schema fields; bin_size scaling.
4. Width-bias + PSF calibration (can follow independently).

## 8. Open questions for review

- **Peak detection dependency.** The estimator assumes a reasonable peak column;
  robust multi-peak detection is issue #7. For now run it on the global-max peak
  (well-posed only for long chords — see `devel/adrt_coordinate_transform.md`).
- **Background/windowing policy** in the accumulator — median vs. fitted pedestal
  vs. matched-filter residual; how much it affects the moment sums on real data.
- **Schema scope.** Do we persist width/length/endpoints and the fit
  diagnostics, or only the refined `(ρ, θ)` + segment? Depends on downstream
  consumers of the streak catalog.
- **Module boundary.** Keep in `adrtDetect.py` or split a dedicated
  `adrtButterfly.py`?
