# ADRT butterfly analysis — implementation plan

Plan to turn the validated closed-form result in [`butterfly.md`](butterfly.md)
into a reusable analysis that estimates line-segment geometry
(**slope/height → ρ,θ; length; width; endpoints**) directly from the ADRT
accumulator around a detected peak. This is the ADRT counterpart of the
Xu–Shin–Klette Hough butterfly, and it slots into the existing `ADRTDetectTask`
between peak finding and post-processing.

- **Status:** steps 1–3 implemented (2026-08-22). The estimator, tests, and task
  wiring are in place; width-bias/PSF calibration (step 6) and robust accumulator
  conditioning remain.
- **Depends on:** the closed-form derivation and inversion in
  [`butterfly.md`](butterfly.md) (validated in `devel/butterfly_closed_form.py`).

## Implementation status

- **Done:** `ADRTSegmentEstimate`, `estimate_segment_adrt`,
  `_column_moments_continuous`, `_invert_inertia` in
  `src/astro_lfd/algorithms/adrtDetect.py`; `_find_peaks` returns integer
  `(q, h, s)` peaks; `detect` runs the estimator and applies `bin_size` scaling
  via `_apply_bin_size`; `postprocess` builds a `LineSegment2D` from
  center/angle/length and persists `line_width` (added to the schema in
  `streakAdapter.py`). Tests in `tests/test_adrt_butterfly.py` (53 cases): pure
  inversion is exact; on clean signal length < 1 %, angle < 0.05°, width within
  the additive bias; the full `run()` path recovers θ=115.000°, length≈1498,
  width≈8.19 on a noise-free exposure.
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
run → preprocess → detect(_find_peaks → _adrt_to_hesse) → postprocess
```

`_find_peaks` today returns only `(rho, theta)` from the global argmax, and
`postprocess` builds a `Line2D` clipped to the bbox — so **length/width are not
estimated at all**; the segment is just the infinite line clipped to the frame.

The butterfly analysis adds a stage that consumes a peak `(q, h, s_idx)` plus a
local accumulator window and returns full segment geometry. It reuses the
already-validated `_adrt_to_hesse` for the continuous-coordinate mapping, so it
introduces no new coordinate math — only moment sums and the closed-form
inversion.

---

## 2. Proposed API

A single pure function plus a small result container, kept independent of the
LSST task machinery (like the existing transform helpers) so it is unit-testable
without the stack:

```python
@dataclass
class ADRTSegmentEstimate:
    rho: float            # Hesse ρ (ADRT pixel grid), from refined peak column
    theta: float          # Hesse θ (rad)
    length: float         # px
    width: float          # px
    center_x: float       # px (ADRT pixel grid)
    center_y: float
    # diagnostics
    A: float; B: float; C: float          # V(s) = A s² + B s + C
    slope: float                          # s0 = tan(phi0)
    var_residual: float                   # goodness of the quadratic fit
    n_columns: int

def estimate_segment_adrt(
    adrt_result: NDArray,     # (4, 2N-1, N)
    q: int, h: float, s_idx: int,   # peak (from _find_peaks)
    N: int,
    *,
    half_band: int = 90,           # slope columns each side of the peak
    background: str = "median",    # per-column background subtraction
    subtract_width_bias: float = 0.0,   # calibrated Δ(w²) in px²
) -> ADRTSegmentEstimate:
    ...
```

Placement: extend `src/astro_lfd/algorithms/adrtDetect.py` (next to
`_adrt_to_hesse`), or a sibling `adrtButterfly.py` if it grows. Lean toward the
same module first — it shares the transform helpers and keeps the ADRT math in
one place.

---

## 3. Algorithm (per detected peak)

1. **Select the slope band.** Columns `s_idx − half_band … s_idx + half_band`,
   clipped to `[1, N−1]` and to the peak's quadrant (do not cross seams).
2. **Map to continuous coordinates.** For each column, map its rows through
   `_adrt_to_hesse(q, h_rows, s_col, N)` → `(ρ, θ)`, then
   `s = −cosθ/sinθ`, `b = ρ/sinθ`. (θ is constant within a column.)
3. **Condition each column.** Subtract a local background (median of the column,
   or a windowed estimate around the ridge) so the moment sums see the segment,
   not the sky pedestal. Optionally restrict to a height window around the ridge
   centroid to limit contamination.
4. **Column moments.** Intensity-weighted centroid `μ(s)` and variance `V(s)`
   (the `column_moments` already prototyped, but in continuous `b`).
5. **Fit.** Weighted least squares: linear `μ(s) = β0 + β1 s`, quadratic
   `V(s) = A s² + B s + C`. Weight by column total flux (SNR) so the peak column
   dominates and far, contaminated columns matter less.
6. **Invert** (closed form from `butterfly.md` §3):
   `φ0 = ½ atan2(−B, A−C)`; `L² = 6[(A+C)+√((A−C)²+B²)]`;
   `w² = 6[(A+C)−√((A−C)²+B²)] − subtract_width_bias`;
   `(x_c, y_c) = (−β1, β0)`.
7. **Refine ρ, θ.** Take θ from the fitted `φ0` (normal angle = φ0 + 90°) and ρ
   from the centroid line through `(x_c, y_c)` — a sub-pixel refinement of the
   integer-peak `_adrt_to_hesse` output.
8. **Package** into `ADRTSegmentEstimate` with fit diagnostics.

`postprocess` then builds a `LineSegment2D` from `(center, φ0, L)` via
`LineSegment2D.from_center_length` (already exists) instead of clipping an
infinite line — giving true endpoints.

---

## 4. New / changed source

- `src/astro_lfd/algorithms/adrtDetect.py`
  - add `ADRTSegmentEstimate`, `estimate_segment_adrt`, and small private
    helpers `_column_moments_continuous`, `_invert_inertia`.
  - `_find_peaks` returns the peak indices `(q, h, s_idx)` (not just ρ,θ) so the
    estimator can run; keep the ρ,θ path for back-compat / fallback.
  - `detect` calls `estimate_segment_adrt` per peak, applies `bin_size` scaling
    to `length`/`width`/`center`/`rho` (all lengths scale by `bin_size`; θ
    unchanged), and passes segments to `postprocess`.
  - `postprocess` consumes segments (center, angle, length, width) and writes
    endpoints + width to the streak catalog.
- `src/astro_lfd/table/streakAdapter.py` — add `width` (and, if not present,
  length/endpoint) fields to the schema so the estimate is recorded. **Check the
  current schema first**; `setLineSegment` may already carry endpoints.

## 5. Tests (`tests/`)

- **Unit (no stack):** synthetic ADRT columns with known `(μ20, μ11, μ02)` →
  assert `estimate_segment_adrt` inverts to the input tensor exactly.
- **Integration (sim, needs stack):** port `devel/butterfly_closed_form.py` into
  a parametrized test over a small `(φ0, L, w)` grid; assert
  `|ΔL|/L < 1 %`, `|Δφ0| < 0.05°`, and `|Δw|` within the calibrated bias
  tolerance (e.g. `< 15 %` at `w ≥ 4`, tighter after bias subtraction).
- **Regression:** width-bias constant `w_hat² − w²` stable vs. `w` (guards the
  Sheppard-correction assumption).

## 6. Calibration work (before trusting width)

- Characterize the additive `w²` bias vs. angle, quadrant, and PSF blur; store a
  small calibration (constant or low-order in angle) and wire it into
  `subtract_width_bias`.
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
