# meas.detectStreaks — KHT streak detection task

**When relevant:** using or extending the KHT streak detector (`KHTDetectTask`),
understanding its config/output, its coordinate convention, or how it relates to
`lsst.meas.algorithms.maskStreaks.MaskStreaksTask`.

**Verified:** imports + light unit tests (`tests/test_detectStreaks.py`) against
the LSST stack `lsst-scipipe-13.0.0`, 2026-07-06. Full task run not yet exercised
(needs a realistic exposure — see `docs/KHT_MASKSTREAKS_TESTPLAN.md`).

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

**See also:** [geom-line](geom-line.md),
`docs/KHT_MASKSTREAKS_DISCREPANCY.md`
