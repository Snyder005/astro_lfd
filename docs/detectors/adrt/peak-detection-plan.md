# ADRT multi-peak detection — implementation plan

Issue: #7 (`feature/issue-7-adrt-peak-detection`). Task statement:
`devel/ADRT_PEAK_DETECTOR_TASK.md`.

## Overview

`ADRTDetectTask._find_peaks()` currently returns the single global argmax of the
`(4, 2N-1, N)` ADRT accumulator. It must become a fast multi-peak detector for
low-SNR streaks that returns `list[Line2D]` in the PIXEL frame, with every
tunable as a keyword argument (not a Config field), and without touching
`preprocess`.

The chosen approach is a four-stage, fully vectorized pipeline on the
**line-length-normalized** accumulator `S = A / sqrt(L)`, where `L = adrt(ones)`
is the digital line length of every accumulator cell. On white noise `S` has
unit variance everywhere (measured std 0.99; per-quadrant `1.4826·MAD` =
0.98–0.99), so a single threshold in sigma units is valid across all slopes and
offsets, which raw `A` (std ∝ sqrt(L)) and mean-normalized `A/L` are not.
The stages are: (1) per-quadrant `scipy.ndimage.maximum_filter` local maxima
above `median + k·MAD`; (2) a **height-axis local-contrast test** that rejects
the broad "butterfly wing" ridges a bright streak casts across neighbouring
slope columns; (3) greedy value-ranked **cone non-maximum suppression** inside
each quadrant, using the measured wing decay envelope; (4) closed-form
`(q, h, s) → (rho, theta)` conversion via a restored `_adrt_to_hesse`, then
Hesse-space de-duplication of the same line found on both sides of a quadrant
seam. Optional sub-pixel refinement is a 3-point parabolic centroid by default,
with the butterfly analysis (`extract_segment_adrt`) as an opt-in.

Validated on N=2048 simulations (unit Gaussian noise, FWHM 3 px): the pipeline
returns exactly the injected streaks — and nothing else — for a bright single
streak (peak 20σ/px), a bright streak crossed by a 1σ/px streak, a 1°-separated
faint parallel pair, three mixed streaks in three quadrants, streaks on the 45°
and 90° seams, a 0.4σ/px streak (13σ in `S`), and pure noise (0 peaks). Cost
after the transform is ~1.3 s at N=2048 and ~5 s at N=4096, dominated by
`maximum_filter`; everything downstream operates on ≤ a few thousand
candidates and is negligible.

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
| --- | --- |
| Detect on `S = A / sqrt(L)` with `L = adrt(ones(N,N))` | Accumulator cell variance under white noise is `σ²·L` and `L` ranges from 1 to N across a quadrant → any threshold on raw `A` is either too loose for long lines or too strict for short ones → dividing by `sqrt(L)` makes `S` a unit-variance significance statistic (measured 0.99), so one `k` in sigma units is correct for every cell. `A/L` (mean per pixel) was rejected because its noise std scales as `1/sqrt(L)` and short, noisy lines dominate. |
| `L` computed by `adrt.adrt(np.ones((N, N)))` and cached per `N` on the task instance | Digital line length is exactly the ADRT of a unit image → no separate geometry code to maintain → cost is one extra transform (2.7 s at N=4096) paid once per array size, not per exposure. `_find_peaks` accepts an optional `line_length` array so a caller can pass `adrt(valid_mask)` instead (see preprocessing recommendations). |
| Per-quadrant statistics and candidate extraction, never on a stitched mosaic | The four quadrants are separate 45° angle bands whose `h` axes have different meanings → a stitched array creates false neighbourhoods across seams and `maximum_filter` footprints would straddle them → treat `axis 0` as a batch dimension (footprint size 1 there) and compute median/MAD per quadrant. |
| Threshold = per-quadrant `median + k·MAD`, `k=6`, estimated on every 8th cell with `L > 0` | Median/MAD are robust to the ridges of real streaks → the estimate is unbiased even when a bright streak is present → subsampling by 8 in both axes (65k cells/quadrant at N=2048) gives the same estimate to <1% at 1/64 of the cost of a full median (4.2 s at N=4096 for the full array). `L == 0` cells are structurally empty corners and would bias the median. `k=6` gives 0 false peaks on pure noise at N=2048 while a 0.4σ/px, 1500-px streak (13σ in `S`) is detected. |
| Local maxima via `scipy.ndimage.maximum_filter(size=(1, 7, 31), mode="nearest")` then `S == max` | The streak ridge in `S` has a half-max footprint of ~5 (h) × 11 (s) cells for a 4-px-wide streak and ~9 × 18 for 8 px → a footprint slightly larger than the ridge yields one maximum per ridge without merging distinct streaks → `(7, 31)` is anisotropic because the ridge is elongated along slope. It is a size, not a `min_distance`, so `peak_local_max`'s extra bookkeeping is unnecessary; the separable rank filter costs 1.1 s at N=2048 / 5 s at N=4096. |
| Height-axis local-contrast test: `S[q,h,s] − median(S[q, h±[12,40), s]) > k_contrast·MAD`, `k_contrast=5` | A bright streak's butterfly wings are ridges that are narrow in `s` but very broad in `h` (they are the image projected at the wrong slope) → a genuine line peak sits on a locally flat background whereas a wing local-maximum sits on a plateau nearly as high as itself → comparing the peak to the median of the same column 12–40 cells away rejects wings before NMS. Measured: 2513 → 191 candidates for a bright single streak, with the true peak always surviving; the annulus is gathered by fancy indexing on the candidate list only, so cost is ~10 ms. |
| Greedy cone NMS in `(h, s)` index space per quadrant, envelope `min(1, 40/Δs)`, cone `Δh ≤ Δs + 4` | The wing of a peak at `(h0, s0)` lies inside the cone `|h−h0| ≤ |s−s0|` (the line's pixels re-project within that band) and its amplitude decays as roughly `1/Δs` (measured ratios 0.87 @10, 0.47–0.8 @20, 0.26–0.45 @50, 0.19–0.3 @80) → suppressing only candidates that are both inside the cone and weaker than `40/Δs` of the accepted peak removes the residual wing maxima while keeping any candidate that is brighter than a wing could be → a faint parallel pair 46 slope columns apart (ratio 0.95) survives, while 190 residual wing candidates of the bright streak do not. Greedy by descending `S` is the standard Hough NMS and gives the ranked output for free. |
| Restore the closed-form `_adrt_to_hesse(q, h, s, N)` instead of `adrt.utils.coord_adrt` lookup tables | `coord_adrt(N)` allocates an `(4, 2N-1, N)` float64 offset array (0.5 GB at N=4096) to look up a handful of cells → the closed-form map (commit `bb0a56d`) reproduces it to 1e-13 and accepts fractional indices → required anyway for sub-pixel peaks, and it is the exact inverse of the existing `_hesse_to_adrt`. Placed in `adrtButterfly.py` beside its inverse so the two coordinate maps and their round-trip test live together. |
| Hesse-space de-duplication after conversion: `|Δθ| < 0.5°` and `|Δρ| < 3 px` (θ wrap handled) | A line exactly on a seam angle (0°, 45°, 90°, 135°) appears with identical value in columns `s=0` or `s=N−1` of two adjacent quadrants → the per-quadrant NMS cannot see across quadrants → de-duplicate in physical coordinates where the two detections coincide. Measured: 45° and 90.2° streaks each produce two seam detections with `Δρ ≤ 4 px`, `Δθ ≤ 0.2°`. |
| Default sub-pixel refinement: 3-point parabolic centroid in `h` and `s` around each accepted peak | `_adrt_to_hesse` is analytic in `(h, s)` → fractional indices propagate directly to `(rho, theta)` → a parabola through the peak and its two neighbours is O(1) per peak and is the standard Hough sub-cell estimate. Butterfly refinement (`refine="butterfly"`) is opt-in because `extract_segment_adrt` does a median over a `(2N−1)×180` band per peak (~0.1 s each at N=4096) and is still under calibration. |
| Exclude cells with `L < min_length` (default 64) | Cells with very short digital lines (image corners) have few pixels → a single hot pixel or cosmic ray gives a large `S` → require a minimum footprint. `min_length` is in pixels of the (binned/padded) grid. |
| `max_peaks` default 50, applied after NMS, ranked by `S` | Bounding the greedy loop bounds runtime on pathological inputs (e.g. a saturated image where thousands of candidates survive) → 50 is far above the expected streak count per LSSTCam amplifier-set image → the list is already sorted, so truncation returns the most significant lines. |
| All parameters are keyword-only arguments of `_find_peaks` with defaults; `detect()` passes none | Task requirement → keeps Config unchanged while the algorithm is being tuned → promote to Config later only once defaults are validated on real data. |
| `_find_peaks` returns `list[Line2D]`; the ADRT indices and `S` values are exposed on a companion `list[AdrtPeak]` via a `return_peaks=True` switch | Task requires `Line2D` for `postprocess` → tests, butterfly refinement and ranking need `(q, h, s, value)` → a small frozen dataclass carrying both avoids recomputing `_hesse_to_adrt` and keeps the default return type simple. |

### Rejected Alternatives

| Alternative | Why Rejected |
| --- | --- |
| `skimage.feature.peak_local_max` on the whole array or per quadrant | Internally runs the same `maximum_filter` plus a Python-level sort and `min_distance` exclusion; isotropic `min_distance` does not fit the anisotropic ridge; adds a `scikit-image` dependency to the `[adrt]` extra for no capability gain. |
| CLEAN-style iterative detect → back-project → subtract → re-transform | Each iteration costs a forward ADRT (2.9 s at N=4096) plus an inverse; for the expected few streaks per image it is 5–10× slower than one-pass NMS and its convergence criteria add parameters. Kept as a possible future verification step (design.md Step 6), not as the detector. |
| `scipy.ndimage.label` connected components above threshold, one peak per component (pyhough approach) | A bright streak's wings connect to its core above any low threshold, so the whole butterfly becomes one component and a faint crossing streak inside the cone is absorbed. The local-contrast test plus cone NMS separates these (validated on the bright+faint cross case). |
| Matched filter along `h` before peak finding (prior plan option A1) | Improves SNR by ~sqrt(ridge width) for wide streaks but blurs the ridge, requiring a larger footprint and shifting the threshold calibration; the un-filtered `S` already detects a 0.4σ/px streak at 13σ. Recorded as a possible input-side enhancement. |
| NMS in Hesse `(rho, theta)` space using the intersection point | First implementation attempt needed a correct treatment of θ wrap and image-bounds intersection and still left 50 wing residuals on the bright single case, whereas the index-space cone with the measured `1/Δs` envelope removed all of them. The index-space geometry is exact for the ADRT's butterfly (the cone is a property of the transform), so it is the simpler and validated choice; Hesse space is used only for the seam de-duplication where index space cannot compare quadrants. |
| Full `np.median(S, axis=1)` per column for background | 4.2 s at N=4096 for a quantity that is constant to <1% across a quadrant under the `sqrt(L)` normalization; a subsampled scalar per quadrant is sufficient. |
| Elevating `k`, footprint, etc. to `ADRTDetectConfig` | Explicitly excluded by the task while the algorithm is being developed. |

### Constraints & Assumptions

- `adrt` 1.2.0 API: `adrt.adrt(img)` requires a square power-of-two image and returns `(4, 2N-1, N)`; quadrant angle bands are `[0,π/4)→q3, [π/4,π/2)→q2, [π/2,3π/4)→q1, [3π/4,π)→q0` (normal-vector θ); columns `s=0` and `s=N-1` are seam slopes shared with the adjacent quadrant.
- PIXEL frame conventions (`devel/pixel_coordinate_system.md`): `arr[j, i] ↔ (x=i, y=j)`; Hesse `x cosθ + y sinθ = ρ`; `Line2D` canonicalizes θ to `[0, π)` (`src/astro_lfd/geom/line.py:691-711`). Padding rows are appended after the last row, so no origin shift is needed; any `bin_size` rescale of `rho` is the caller's responsibility, as today.
- Noise model assumption: the ADRT input is approximately white with a spatially uniform variance (i.e. `preprocess` will eventually supply `D/sqrt(V)`). If the variance varies strongly across the image, `S` is still monotone in significance but the single per-quadrant threshold becomes approximate.
- Dependencies: `numpy`, `scipy.ndimage` (already core deps), `adrt` (already the `[adrt]` extra). No new packages.
- Code style: black line length 110, numpydoc docstrings, mypy py313, `lsst.pipe.base.Task` pattern; tests use `pytest.importorskip` for stack-dependent modules and follow `tests/test_adrt_butterfly.py` (simulate with `astro_lfd.sims.Streak.from_center_length`, locate truth with `_hesse_to_adrt`).
- Git workflow: work on `feature/issue-7-adrt-peak-detection`, small conventional commits, draft PR referencing `Fixes #7`, no auto-merge.
- Measured timings (this SDF node): `maximum_filter(size=(1,7,31))` 1.1 s at N=2048 and ~5 s at N=4096; `adrt(ones)` 2.7 s at N=4096; the remaining stages < 50 ms.

### Known Risks

| Risk | Mitigation | Anchor |
| --- | --- | --- |
| `detect()` treats the `_find_peaks` result as an array (`result.size`), which fails for a `list` | Milestone 2 rewrites `detect()` to use `len(lines)`. | `src/astro_lfd/algorithms/adrtDetect.py:139-141` — `result = self._find_peaks(adrt_result)`; `f"... {result.size:d} line(s)"`; `return [] if result.size == 0 else result` |
| `preprocess()` and `postprocess()` are broken after the refactor in `16ae4b8` (`mi`, `self.config.bin_size`, `segments`, `shape`, `streak_center`, `detected_mask`, `self.config.streak_width`, `self.config.only_mask_detected` are undefined), so `run()` cannot be exercised end-to-end | Out of scope per the task ("do not make changes to the preprocessing"); tests call `_find_peaks` directly on simulated arrays. Flagged for a separate issue/PR; the plan lists the recommended `preprocess` changes below. | `adrtDetect.py:99` `get_pixel_mask(mi.mask, ...)`; `:110` `4096 // self.config.bin_size`; `:168` `if not segments:`; `:173` `np.zeros(shape, ...)`; `:189` `self.config.streak_width`; `:196-197` `self.config.only_mask_detected`, `detected_mask` |
| `__all__` exports names that no longer live in the module | Milestone 2 updates `__all__` to the two remaining public names. | `adrtDetect.py:1` — `__all__ = ["ADRTDetectConfig", "ADRTDetectTask", "ADRTSegment", "extract_segment_adrt"]` |
| A faint streak within ~1° of a much brighter one, crossing it inside the image, is suppressed by the cone envelope if its `S` is below `40/Δs` of the bright peak | Accepted: this is the regime where the wing genuinely dominates the faint ridge; `wing_scale` is exposed as an argument so a caller can tighten it. Documented in the README. | — |
| Real-data noise is not white (sky gradients, PSF-correlated noise, residual objects), so the `k=6` calibration may under- or over-detect | `k`, `k_contrast`, `min_length` are arguments; per-quadrant MAD adapts the scale automatically. Recommend evaluating on the `testdata` images and real LSSTCam exposures in a follow-up before promoting defaults to Config. | — |
| Structurally empty corner cells (`L == 0`) would give `S = 0/0` | `S` is computed with `L` clipped to ≥ 1, and those cells are also excluded by `min_length`. | — |
| Butterfly refinement uses a slope band that is clipped to the quadrant interior and can raise `ValueError` for peaks at `s ≤ 1` or `s ≥ N−2` | With `refine="butterfly"`, catch `ValueError` per peak and fall back to the parabolic estimate, logging at debug level. | `src/astro_lfd/algorithms/adrtButterfly.py:319-320` — `lo = max(1, s_idx - half_band)`; `hi = min(N - 1, s_idx + half_band)` |
| `maximum_filter` on a float64 `(4, 8191, 4096)` array is the dominant cost (5 s) and memory (0.5 GB for `S` plus 0.5 GB for the filtered copy) | Accepted for now; compute `S` in float32 (halves both) — the statistic does not need float64 precision. Note in README that binning (`bin_size=2`) reduces this 4×. | — |

## Invisible Knowledge

To be captured in `src/astro_lfd/algorithms/README.md` (ADRT section):

1. **Why `A / sqrt(L)`.** Each accumulator cell is a sum over `L` pixels; under white noise of variance σ² its variance is `σ²·L`. Dividing by `sqrt(L)` makes every cell a z-score, so a single threshold in sigma units is valid across slopes and offsets. `L` is obtained by transforming an all-ones image (or the valid-pixel mask): the ADRT of the indicator is the digital line length.
2. **The butterfly is the enemy of multi-peak detection.** A bright streak does not produce a single peak but a "butterfly": the true peak plus two wings that fan out across slope columns inside the cone `|Δh| ≤ |Δs|`, with amplitude decaying roughly as `1/Δs`. Wings are broad along `h` (they are the streak re-projected at the wrong angle), which is why a height-axis local-contrast test separates a wing from a genuine line peak, and why the NMS uses a cone rather than a box.
3. **Quadrants are separate images.** The `(4, 2N-1, N)` array is four independent 45° bands; neighbourhood operations must never cross axis 0, and a line on a seam angle appears in two quadrants at once. De-duplication must therefore happen in physical `(rho, theta)` coordinates, after conversion.
4. **Coordinate maps.** `_adrt_to_hesse` and `_hesse_to_adrt` are exact inverses of each other and of `adrt.utils.coord_adrt` (to 1e-13); the forward map accepts fractional `(h, s)` so sub-pixel peaks convert directly. Both operate on the padded/binned grid; the caller undoes binning by scaling `rho`.
5. **Performance profile.** After the forward transform, the detector is one separable rank filter over the full array (~5 s at N=4096) plus operations on the candidate list only; median/MAD are estimated on a 1/64 subsample. Peak count is bounded by `max_peaks` so the greedy loop is O(max_peaks × candidates).
6. **Ranking.** Output lines are sorted by descending `S` (significance), not by raw accumulator value; a long faint streak can outrank a short bright one.
7. **Parameter placement.** The tunables are method arguments by design while the algorithm is tuned on simulations; promotion to `ADRTDetectConfig` is deferred until real-data defaults are established.

### Recommendations for the ADRT input (not implemented here)

- Feed `D / sqrt(V)` (image over sqrt variance) so the white-noise unit-variance assumption behind the `k` threshold holds; with `V` available, `L` should be the ADRT of the valid-pixel mask (`~bad_mask`), which also makes `S` exact for masked regions. `_find_peaks` already accepts this via `line_length`.
- Keep `bin_size` (2×2 median or sum) as a performance lever: it divides the filter cost and memory by 4 and slightly increases per-cell SNR for streaks wider than 2 px.
- Restore `preprocess`/`postprocess` consistency (see Known Risks) in a dedicated PR; `postprocess` needs the image `shape`, a `streak_width` source (the butterfly width or a config default), and the `detected_mask` plumbing.
- Optionally smooth the input with the PSF (matched filter) before the transform when targeting very faint, wide streaks.

## Milestones

### Milestone 1: Restore the closed-form ADRT → Hesse converter

**Files**: `src/astro_lfd/algorithms/adrtButterfly.py`, `tests/test_adrt_butterfly.py`

**Code Intent**:

- Add `_adrt_to_hesse(q, h, s, N) -> tuple[NDArray, NDArray]` next to `_hesse_to_adrt`, vectorized over array inputs, accepting fractional `h` and `s`, returning `(rho, theta)` in the padded grid's PIXEL frame with θ reduced to `[0, π)` and `rho` sign-flipped accordingly (so it matches `Line2D` canonicalization and `_hesse_to_adrt`'s input convention). Reuse the derivation from commit `bb0a56d` (Decision Log: "Restore the closed-form converter").
- Docstring documents the quadrant/angle bands and the seam degeneracy, mirroring `_hesse_to_adrt`.
- Tests: (a) round-trip `_hesse_to_adrt ∘ _adrt_to_hesse` on a grid of integer `(q, h, s)` for N=64 and N=256 recovers indices to 1e-9; (b) agreement with `adrt.utils.coord_adrt(N)` offset/angle arrays after mapping to Hesse form, to 1e-10; (c) a fractional-index case is continuous (midpoint between two cells maps between their lines).

### Milestone 2: Implement `_find_peaks` and repair `detect`

**Files**: `src/astro_lfd/algorithms/adrtDetect.py`

**Code Intent**:

- Add a frozen dataclass `AdrtPeak(q, h, s, value, line)` (integer indices of the accepted local maximum, refined fractional `h`/`s` optional fields, significance `value`, and the `Line2D`).
- Rewrite `_find_peaks(self, adrt_result, *, line_length=None, k=6.0, k_contrast=5.0, footprint=(7, 31), contrast_annulus=(12, 40), wing_scale=40.0, cone_pad=4, min_length=64, max_peaks=50, dedup_rho=3.0, dedup_theta_deg=0.5, refine="parabolic", return_peaks=False) -> list[Line2D] | tuple[list[Line2D], list[AdrtPeak]]` implementing, in order:
  1. Derive `N` from the array shape; obtain `L` from `line_length` or from a per-instance cache keyed by `N` (`adrt.adrt(np.ones((N, N)))`); form `S = A / sqrt(max(L, 1))` in float32.
  2. Per quadrant: robust `median`/`MAD` on the `[::8, ::8]` subsample restricted to `L > 0`; candidates = `(S == maximum_filter(S, size=(1, *footprint), mode="nearest")) & (S > median + k·MAD) & (L ≥ min_length)`.
  3. Height-axis local-contrast test on the candidate list using the annulus offsets, keeping `S − median(annulus) > k_contrast·MAD_q`.
  4. Sort by descending `S`; greedy cone NMS within each quadrant with envelope `min(1, wing_scale/Δs)` and cone `Δh ≤ Δs + cone_pad`; stop at `max_peaks`.
  5. Sub-pixel refinement per `refine`: `"none"`, `"parabolic"` (3-point fit along `h` and `s`, clipped to ±0.5 cell), or `"butterfly"` (call `extract_segment_adrt`, fall back to parabolic on `ValueError`).
  6. Convert with `_adrt_to_hesse`, build `Line2D(rho, theta * geom.radians)`, then Hesse-space de-duplication (θ-wrap aware) keeping the higher-`S` member.
  7. Log a debug summary (candidates per stage) and return.
- Rewrite `detect()` to call `_find_peaks(adrt_result)`, log `len(lines)`, and return the list; update the docstrings to the new behaviour; fix `__all__` to `["ADRTDetectConfig", "ADRTDetectTask"]`. Leave `preprocess`/`postprocess` untouched (Known Risks).

### Milestone 3: Simulation tests for multi-peak detection

**Files**: `tests/test_adrt_peaks.py`

**Code Intent**:

- Module-level fixtures: N=1024 (fast) unit-noise image generator with a fixed seed; streaks built with `Streak.from_center_length(...)`, `get_signal(shape, fwhm=3.0)`; truth indices via `_hesse_to_adrt`; a helper that asserts a detected `Line2D` matches a `Streak` within `|Δρ| ≤ 3 px`, `|Δθ| ≤ 0.3°`.
- Cases (each asserts exact count and matching, and that ranking follows injected brightness where unambiguous): bright single streak → 1 line; bright + faint crossing streak in another quadrant → 2; faint parallel pair 1° apart → 2; three streaks in three quadrants → 3; streak at 45° and at 90.2° (seams) → 1 each; very faint streak (0.4σ/px) → 1; pure noise → 0.
- Argument tests: `max_peaks=1` returns the brightest; `return_peaks=True` yields `AdrtPeak` objects consistent with the lines; `refine="butterfly"` returns within the same tolerance as parabolic; `line_length` supplied explicitly equals the cached path.
- Mark the N=2048 timing check as `slow`/skipped by default.

### Milestone 4: Documentation and knowledge capture

**Files**: `src/astro_lfd/algorithms/README.md` (new or extended), `docs/detectors/adrt/design.md` (Step 4/5 updated to reference the implementation), `knowledge/adrt-peaks.md` + `knowledge/INDEX.md`, `devel/ADRT_PEAK_DETECTOR_TASK.md` (mark status)

**Code Intent**:

- README: the Invisible Knowledge items above, the argument table with defaults and units, and the preprocessing recommendations.
- design.md: replace the sketch in Steps 4–5 with a pointer to `_find_peaks` and the measured calibration numbers.
- knowledge note: one-screen summary (statistic, stages, defaults, validated cases, cost) with hook in `INDEX.md`.
- Open a draft PR (`Fixes #7`) after Milestone 2; mark ready after Milestones 3–4 pass `pytest tests/test_adrt_peaks.py tests/test_adrt_butterfly.py`.
