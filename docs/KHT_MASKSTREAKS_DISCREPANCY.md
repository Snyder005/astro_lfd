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
