# Discrepancy analysis: KHTDetectTask vs. MaskStreaksTask

**Date:** 2026-07-06
**Method:** static side-by-side reading of
`astro_lfd.meas.detectStreaks.KHTDetectTask` and
`lsst.meas.algorithms.maskStreaks.MaskStreaksTask` (upstream `main`). No full
task run — empirical confirmation is deferred to `KHT_MASKSTREAKS_TESTPLAN.md`.

## Scope and framing

Both tasks share the **same detection engine**: Canny edge extraction
(`skimage.canny`, `sigma=0.1`, `use_quantiles`), `lsst.kht.find_lines` with the
same parameters, recursive-KMeans clustering with the same stop rule, and — most
importantly — the **same profile fitter** (`Line`, `LineProfile`), which
`KHTDetectTask` *imports from* `maskStreaks` rather than reimplementing. So the
Newton–Raphson Moffat fit itself is **not** a discrepancy source.

Differences are therefore concentrated (a) **before** the fit, in how the edge /
weight images are built, and (b) **after** the fit, in how the fitted `(rho,
theta)` is expressed, canonicalized, consolidated, and accepted into the output.
The two also emit fundamentally different products — `MaskStreaksTask` returns a
`LineCollection` + union `mask`; `KHTDetectTask` returns a `SourceCatalog` of
`StreakAdapter` segments + per-line `Footprint`s — so "compare the lines" means
comparing rho/theta after mapping to a common frame.

Ranked below, highest expected contribution first.

---

## 1. Coordinate frame / origin of the output lines — *largest*

`LineProfile` fits in an image-array frame **centered on the image**
(`xrange = arange(xmax) - xmax/2`, `yrange = arange(ymax) - ymax/2`), with
`theta` in **degrees**. `MaskStreaksTask` returns those center-relative `Line`s
**verbatim** in `.lines`.

`KHTDetectTask` instead re-expresses each fit as
`Line2D(fit.rho, fit.theta * geom.degrees).translated(Extent2D(box.getCenter()))`
and clips to `box`, i.e. it maps into an **absolute pixel frame**. Three offsets
enter here and are the dominant source of raw-number disagreement:

- **center vs. corner:** `+ (nx/2, ny/2)` translation applied by `KHTDetectTask`
  but not by `MaskStreaksTask`.
- **`XY0`:** `box = exposure.getBBox()` starts at the exposure's `XY0`, which is
  `(0,0)` for the synthetic testdata but **non-zero** for a sub-image / calexp.
- **bbox choice:** `KHTDetectTask` uses `exposure.getBBox()` (matches the fitted
  array). The original prototype used `detector.getBBox()`; if a future variant
  reintroduces that, a detector-vs-image bbox mismatch would add another offset.

**Check:** shift one side by `±bboxCenter` (and account for `XY0`) before
matching; residual `Δrho` should collapse to ~0 for shared detections.
**Impact:** without alignment, `rho` differs by hundreds/thousands of pixels;
with alignment, ~0. This is a representation difference, not a detection error.

## 2. Angle canonicalization and rho sign — *high*

`Line2D.__init__` runs `_canonicalize`: `theta` is folded into `[0, π)` and
`rho`'s sign is flipped when π is subtracted. `MaskStreaksTask` applies **no**
such canonicalization — its `theta` stays in raw degrees and can be negative or
≥ 180.

Consequences when comparing:
- The *same* geometric line can read as `(rho, theta)` in one task and
  `(-rho, theta ± 180°)` in the other. Naive numeric comparison flags a false
  discrepancy.
- Two KHT/cluster lines that are 180° apart are **distinct** rows in a
  `MaskStreaksTask` `LineCollection` but **collapse to one** representation under
  `Line2D` — so the consolidated *counts* can differ even with identical inputs.

**Check:** canonicalize both to `[0, π)` (degrees→radians) before matching;
compare `rho` with a sign-aware/absolute tolerance.

## 3. Pre-KHT masking and dilation differences — *high (affects which lines exist)*

The edge image fed to `lsst.kht` is built differently, so the *set* of detected
lines (upstream of the shared fit) can diverge:

- **Bad-plane set:** `KHTDetectTask` default `bad_mask_planes` adds **`ITL_DIP`**
  and **`SPIKE`** to the `maskStreaks` default (`NO_DATA, INTRP, BAD, SAT,
  EDGE`). More masked area → fewer/shorter edges in those regions.
- **Dilation mechanism:** `KHTDetectTask` uses `scipy.ndimage.
  distance_transform_edt` (a Euclidean-radius dilation of the boolean array),
  applying radius **1** to the whole bad mask and `saturated_detections_dilation`
  (250) to `SAT ∩ DETECTED`. `MaskStreaksTask` dilates per **`SpanSet`**
  (`SpanSet.fromMask(...).split()` then `sset.dilated(r)`) and writes into a
  cloned mask. EDT-radius vs. SpanSet-morphological dilation differ subtly at
  boundaries/corners → slightly different invalid regions → different Canny edges.
- **Weight masking:** `KHTDetectTask` zeroes fit weights on `bad_mask` (the
  1-pixel-dilated bad planes); `MaskStreaksTask` zeroes weights on the
  **undilated** `badMaskPlanes`. Small differences in which pixels drive the fit.

**Check:** align the bad-plane lists and compare edge images pixelwise; count
KHT `originalLines` from each before clustering.
**Impact:** can add or drop whole lines near masked/saturated features, and
nudge fitted parameters via the weight map.

## 4. KMeans clustering non-determinism — *medium (jitter, not bias)*

Both share the identical recursive-KMeans consolidation (rescale by bin sizes →
increase `n_clusters` until every cluster's per-axis std ≤ 1). `KMeans` uses
random initialization (`n_init="auto"`, no fixed `random_state` in either), so
cluster **centers jitter run-to-run** even on identical input. This produces
small `(Δrho, Δtheta)` differences that are **not** attributable to the
reformulation.

**Check:** run each task ≥3× on the same input and measure within-task variance;
only differences exceeding that floor are real. Optionally fix `random_state`.

## 5. Output content / fields carried — *medium (matching & downstream)*

`maskStreaks.Line` carries `sigma`, `reducedChi2`, and `modelMaximum`;
`MaskStreaksTask` sets `modelMaximum` on each accepted line.
`KHTDetectTask` stores only geometry (`line_rho`, `line_theta`, `line_u_center`,
`line_length`, `line_center_*`) + `Footprint` + sky `coord` via `StreakAdapter`;
it **discards** `reducedChi2` / `modelMaximum` / `sigma`.

**Consequence:** any quality-based matching, ranking, or filtering that relies on
chi² or model peak cannot be reproduced from the `KHTDetectTask` catalog — a
comparison gap rather than a geometric one. Consider adding these fields to the
adapter schema if downstream selection needs them.

## 6. Line acceptance and footprint-vs-mask semantics — *low/medium (margins)*

Post-fit acceptance is close but not identical:

- Both drop a line if the fit moves > `2·binSize` in rho/theta, and both apply
  the `footprint_threshold` on `|model|`. Same logic.
- `MaskStreaksTask` then unions all accepted line masks into one `mask` and (with
  `onlyMaskDetected=True`, the default) intersects it with `DETECTED`.
  `KHTDetectTask` has **no** `onlyMaskDetected` step and instead stores a
  **per-line** `Footprint` (`SpanSet.fromMask(|model|>threshold)`) and a canonical
  `LineSegment2D` clipped to the bbox.
- `KHTDetectTask` additionally skips a line when `det_line.intersection(box) is
  None` (line misses the box) — an extra guard absent from the reference's line
  list (it only affects masking there).

**Consequence:** membership can differ by a line or two at the margins (a line
whose footprint survives thresholding but whose segment misses the bbox, or a
faint streak retained by one acceptance path and not the other). The *geometry*
of jointly-accepted lines is unaffected.

---

## Summary ranking

| # | Cause | Effect on comparison | Fix / control |
|---|---|---|---|
| 1 | Output frame/origin (center vs. corner, `XY0`, bbox) | Large raw `Δrho`; vanishes after alignment | Map both to one frame before matching |
| 2 | `Line2D` canonicalization + rho sign | False mismatches; count merges at 180° | Canonicalize both to `[0,π)` |
| 3 | Bad-plane set + EDT vs. SpanSet dilation | Adds/drops lines; nudges weights | Align plane lists; diff edge images |
| 4 | KMeans random init | Small run-to-run jitter | Repeat runs; fix `random_state` |
| 5 | Missing `reducedChi2`/`modelMaximum` | Cannot reproduce quality-based selection | Extend adapter schema if needed |
| 6 | Acceptance / footprint vs. union-mask | Marginal membership differences | Compare with `onlyMaskDetected` off |

**Bottom line:** the reformulation should be *geometrically equivalent* for
jointly-detected streaks once #1 and #2 are reconciled; the remaining
differences (#3–#6) affect *which* streaks appear and *what metadata* is carried,
not the fitted line of a shared detection. Empirical confirmation follows the
harness in `KHT_MASKSTREAKS_TESTPLAN.md`.

---

## Empirical results (harness run 2026-07-07)

**Harness:** `scripts/kht_maskstreaks_compare.py`. Runs `MaskStreaksTask.find`
and `KHTDetectTask.run` on the *same* exposure, maps both outputs into the
absolute-pixel `theta∈[0,π)` frame (reference `Line` wrapped in the same
`Line2D(rho,θ°).translated(bboxCenter)` transform KHT applies to its own fit,
so #1 and #2 are reconciled identically for both sides), matches by `(rho,
theta)` within `(rho_bin_size=40 px, theta_bin_size=2°)`, and reports residuals
plus a KMeans jitter floor. Configs were aligned (shared numeric knobs equal;
`bad_mask_planes` set to the reference `NO_DATA, INTRP, BAD, SAT, EDGE` — i.e.
`ITL_DIP`/`SPIKE` dropped, per review §3). KMeans `random_state` was pinned to 0
for the matched run via a harness-local monkeypatch (no task code changed).

**Inputs:** (A) synthetic `testdata` exposure (θ=30°, ρ=1500, one streak, a
`DETECTED` plane thresholded from the noise-free signal at 5σ); (B) two real
`difference_image`s (visit `2025071700631`, detectors 140 & 136), each with a
populated `STREAK`/`DETECTED`/`SAT` mask and one high-SNR streak.

| Input | n_ref | n_kht | matched | max \|Δρ\| (px) | max \|Δθ\| (deg) | max \|Δlen\| (px) | KMeans jitter |
|---|---|---|---|---|---|---|---|
| A: testdata (seed 12345) | 1 | 1 | 1 | **0.0000** | **0.0000** | 0.0000 | 0 |
| B: diffim det 140 | 1 | 1 | 1 | 0.0041 | 0.00008 | 0.0013 | 0 |
| B: diffim det 136 | 1 | 1 | 1 | 0.0102 | 0.00008 | 0.0034 | 0 |

**Count agreement:** perfect on all three inputs — same number of lines, all
matched, zero unmatched on either side. #6 (acceptance / footprint-vs-mask) did
**not** manifest as a membership difference here (single clean streak per frame;
would need a multi-streak or margin case to exercise).

**Jitter floor (#4):** with `n_init="auto"` and *no* seed, three repeat runs of
each task gave identical line counts and `max|Δρ|=0`, `max|Δθ|=0` within each
task on all inputs. On these single-streak frames the recursive-KMeans stop rule
converges to the same centers every time, so the jitter floor is effectively
zero — every residual below is real, not jitter. (A denser field could still
jitter; the control remains worthwhile.)

**#3 is the sole driver of the residual — confirmed by isolation.** On synthetic
data (no bad pixels near the streak) the residual is *exactly* 0. On real data it
is small but non-zero. Patching KHT's weight/edge bad-mask dilation to the
reference's **undilated** behaviour (radius-1 → identity; SAT dilation of 250
left intact) collapsed det-140's residual from `Δρ=0.00406` to **`Δρ=0.00000`,
`Δθ=0.000000`**. Because the line *count* was unchanged (edges did not gain/lose
a detection), the effect is entirely through the **fit weights**: KHT zeros
weights on the **1-px-dilated** `bad_mask`, whereas `maskStreaks._fitProfile`
zeros them on the **undilated** `badMask` (`maskStreaks.py:891–894`). This is the
weight-masking bug flagged in review §3.

### Re-ranked conclusions

- **#1, #2 (frame/canonicalization):** confirmed pure representation. Once both
  sides use the same translate + `[0,π)` fold, matched `Δρ/Δθ` are ~0. Not bugs.
- **#3 (weight masking):** **the** substantive code difference. Small on these
  frames (< 0.01 px) but a genuine, avoidable divergence in the shared fit; it
  will grow wherever more bad pixels sit within `nSigmaMask` of a streak. **Fix
  recommended** (align KHT fit weights to the *undilated* bad planes). The
  bad-plane *set* (`ITL_DIP`/`SPIKE`) and EDT-vs-SpanSet dilation *shape* were
  neutralized here by aligning the lists; they remain latent count-level
  differences worth quantifying separately if a frame exercises them.
- **#4 (KMeans):** no measurable jitter on these inputs; `random_state` pinning
  worked. Native intent is `n_init="auto"` unseeded — see proposed API note below.
- **#5 (missing fields):** unchanged — KHT still discards `sigma`,`reducedChi2`,
  `modelMaximum`. No geometric impact; blocks quality-based reproduction.
- **#6 (acceptance/footprint):** not triggered by single-streak frames.

**Overall:** the reformulation is **geometrically equivalent** to `MaskStreaksTask`
for jointly-detected streaks — matched lines agree to < 0.01 px / < 0.0001° on
real data and *exactly* on synthetic — with the residual fully explained by the
§3 weight-dilation difference. Reproduce with:

```bash
python scripts/kht_maskstreaks_compare.py --source testdata
python scripts/kht_maskstreaks_compare.py --source butler --detector 140
python scripts/kht_maskstreaks_compare.py --source butler --detector 136
```
