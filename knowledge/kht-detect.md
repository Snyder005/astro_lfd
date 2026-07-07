# meas.detectStreaks — KHT streak detection task

**When relevant:** using or extending the KHT streak detector (`KHTDetectTask`),
understanding its config/output, its coordinate convention, or how it relates to
`lsst.meas.algorithms.maskStreaks.MaskStreaksTask`.

**Verified:** unit tests (`tests/test_detectStreaks.py`) + full end-to-end runs
on synthetic testdata and two real diffims (visit `2025071700631` det 140/136)
against the LSST stack `lsst-scipipe-13.0.0`, 2026-07-07. The task is
**validated against `MaskStreaksTask`**: jointly-detected lines agree to exactly
0 in rho/theta once the two fixes below are in place (see
`docs/KHT_MASKSTREAKS_DISCREPANCY.md` "Empirical results" and the harness
`scripts/kht_maskstreaks_compare.py`).

## What it is

`astro_lfd.meas.detectStreaks.KHTDetectTask` (a `pipeBase.Task`, **not** a
PipelineTask) ports the MixCOATL `detectStreaks` prototype. It reproduces the
detection stages of `MaskStreaksTask` but emits a `SourceCatalog` of line
segments instead of a `STREAK` mask plane. It is the template for future
`astro_lfd` detectors (incl. the ADRT one).

Pipeline: `DETECTED` plane → Canny edges (`skimage`, `sigma=0.1`,
`use_quantiles`) → remove invalid (dilated bad planes + dilated SAT∩DETECTED) →
`lsst.kht.find_lines` → recursive-KMeans clustering (`_cluster_lines`) → per
cluster: `maskStreaks.LineProfile` Moffat fit → accept → `StreakAdapter` row.

## API

- `run(table, exposure) -> Struct(streaks: SourceCatalog, edges: ndarray)`.
  `table` must carry the streak `line_*` schema — build it with
  `StreakAdapter.makeMinimalSchema()`. Early-returns `streaks`+`edges` when KHT
  finds nothing.
- Reuses `Line`, `LineProfile` **imported from** `maskStreaks` — the profile fit
  itself is therefore identical between the two tasks (differences live pre-fit
  in edges/masking and post-fit in frame/canonicalization).
- Helpers (module-level): `get_pixel_mask(mask, plane|list)`,
  `binary_dilation(bool_img, radius)` (scipy `distance_transform_edt` — radius is
  Euclidean, so diagonal neighbors need radius ≥ √2).

## Output fields written per accepted line (via `StreakAdapter`)

Geometry `line_rho`, `line_theta` (`lsst.geom.Angle`), `line_u_center`,
`line_length`, `line_center_{x,y}`; profile-fit quality `line_sigma`,
`line_reduced_chi2`, `line_model_maximum` (added 2026-07-07 to mirror the
`maskStreaks.Line` fields — `model_maximum = abs(finalModel).max()`); a
`Footprint` (`SpanSet` where `|model| > footprint_threshold`); and sky `coord`
(only if the exposure has a WCS). The three quality fields match `maskStreaks`
exactly on shared detections.

## Coordinate convention — READ THIS

`LineProfile` fits `(rho, theta)` in an **image-array frame centered on the
image** (`xrange = arange(nx) - nx/2`, theta in **degrees**). To get absolute
pixel coordinates the task does:
`Line2D(fit.rho, fit.theta*geom.degrees).translated(Extent2D(bbox.getCenter()))`
then `.intersection(bbox)` for the finite segment. It uses `exposure.getBBox()`
(not `detector.getBBox()`) so the frame matches the fitted array and there is no
hard dependence on an attached detector. `Line2D` also **canonicalizes** theta to
`[0,π)` (flipping rho's sign), unlike the raw `maskStreaks` `Line`.

## Gotchas

- `getWcs()`/detector may be absent on synthetic exposures — the task guards WCS
  (`setCoord` skipped when `None`) and uses the image bbox for the frame shift.
- `intersection(box)` can return `None` (line misses the box) — skip those.
- Config fields are **snake_case** (`rho_bin_size`, …) and `bad_mask_planes`
  defaults include extra `ITL_DIP`, `SPIKE` vs the smaller `maskStreaks` set.
- **Two masks, two uses — do not conflate (fixed 2026-07-07).** The bad planes
  feed *two* things: the Canny **edge** mask (one-pixel-dilated, to also drop the
  borders of bad regions) and the **fit weights** (zeroed on the *undilated* bad
  planes, matching `maskStreaks._fitProfile`). An earlier version zeroed weights
  on the dilated mask, which shifted the fit by ~0.004–0.01 px on real data. Keep
  `bad_mask` (undilated → weights) and `dilated_bad_mask` (→ edges) separate. The
  ADRT detector inherits the same weight-masking rule.

**See also:** [geom-line](geom-line.md), [detector-task](detector-task.md),
`docs/KHT_MASKSTREAKS_DISCREPANCY.md`
