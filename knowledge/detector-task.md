# LFD detector-task template (KHT today, ADRT next)

**When relevant:** building the ADRT detector task, or any new `astro_lfd`
line detector — the shared task formulation, input contract, output format, and
coordinate handling that stay the same when only the *line-finding core* changes.

**Verified:** distilled from the validated `KHTDetectTask`
(`knowledge/kht-detect.md`) and its comparison against `MaskStreaksTask`,
`lsst-scipipe-13.0.0`, 2026-07-07.

## The pattern

Every LFD detector is a plain `lsst.pipe.base.Task` (**not** a PipelineTask —
no Butler I/O in the task) with:

```python
run(table, exposure) -> pipeBase.Struct(streaks=SourceCatalog, ...)
```

The KHT pipeline, with the **swappable core marked**:

1. Read `image`, `variance`, `mask` from the `ExposureF`; make the output
   `SourceCatalog(table)`; grab `wcs = exposure.getWcs()` (may be `None`).
2. `detected_mask = get_pixel_mask(mask, config.detected_mask_plane)`.
3. **[CORE — replace for ADRT]** Canny edges → remove invalid regions →
   `lsst.kht.find_lines` → recursive-KMeans clustering → list of candidate
   `(rho, theta)` lines in the **image-centered** frame.
4. Build fit weights (see contract below).
5. Per candidate: `maskStreaks.LineProfile` Moffat fit → acceptance checks →
   map to absolute pixels + canonicalize → write a `StreakAdapter` row.

The ADRT detector keeps **steps 1–2, 4–5 verbatim** and replaces **step 3**:
Canny+KHT → `adrt.adrt(...)` on the significance image + peak-finding +
coord mapping (see [adrt-api](adrt-api.md), [adrt-geometry] once written). The
profile fit, output format, and frame handling are shared, so the ADRT lines
should be directly comparable to KHT/`maskStreaks` lines with the same harness
(`scripts/kht_maskstreaks_compare.py`).

## Input contract (what `run` consumes)

An `lsst.afw.image.ExposureF` with:
- **`DETECTED`** mask plane flagging the streak pixels (the detection seed —
  without it the KHT core returns nothing; the ADRT core should likewise operate
  on detected/significant pixels).
- **Bad planes** present in the mask dict (`NO_DATA, INTRP, BAD, SAT, EDGE`; KHT
  adds `ITL_DIP, SPIKE` by default — note real diffims may lack `ITL_DIP`).
- Finite `image` and `variance` (`variance > 0`; the fit uses `1/variance`).
- Optional WCS (only used for `setCoord`; guarded when `None`).

Real inputs are **difference images** from the Butler (`difference_image`
dataset). Synthetic inputs come from `astro_lfd.utils.testdata` — but its `mask`
starts empty, so a `DETECTED` plane must be added (e.g. threshold the noise-free
signal). See [testdata](testdata.md); note the default `(4004, 4096)` shape is
**not** ADRT-ready (pad to pow2-square first — the ADRT front-end's extra step).

## Weights (shared rule — get this right)

```python
weights = variance**-1
weights[~isfinite(weights) | ~isfinite(image)] = 0
weights[bad_mask] = 0          # UNDILATED bad planes, not the dilated edge mask
```

The bad planes are dilated by 1 px only for the **edge/detection** mask, never
for the weights — this matches `maskStreaks._fitProfile` and was the one real
discrepancy found (see [kht-detect](kht-detect.md) gotchas). The ADRT detector
must use the same undilated-weights rule so its fits stay comparable.

## Output format (what `run` produces)

A `SourceCatalog` built from `StreakAdapter.makeMinimalSchema()`, one record per
accepted line, written through `StreakAdapter` (`streak[key] = value` or the
setters). Fields (see [geom-line](geom-line.md), [kht-detect](kht-detect.md)):

- Geometry: `line_rho`, `line_theta` (`lsst.geom.Angle`), `line_u_center`,
  `line_length`, `line_center_{x,y}` — in the **absolute pixel** frame,
  `theta ∈ [0, π)`.
- Quality: `line_sigma`, `line_reduced_chi2`, `line_model_maximum`.
- A per-line `Footprint` (SpanSet where `|model| > footprint_threshold`); sky
  `coord` if a WCS is present.

This replaces the `maskStreaks` products (a union `STREAK` mask + a
`LineCollection`): the new-style output is one segment per line in standard
detector (PIXELS-frame) coordinates, separable per-line via the `Footprint`.

## Coordinate handling (shared — see kht-detect for detail)

The profile fit works in an **image-array frame centered on the image**
(`arange(n) - n/2`, theta in **degrees**). Map to absolute pixels with
`Line2D(rho, theta*geom.degrees).translated(Extent2D(bbox.getCenter()))` then
`.intersection(bbox)`. Use `exposure.getBBox()` (not `detector.getBBox()`).
`Line2D` canonicalizes theta to `[0, π)`. Any detector whose core emits lines in
a *different* frame (the ADRT (offset, angle) space) must convert into this same
image-centered `(rho, theta[deg])` convention before handing them to
`LineProfile`, so the shared fit + output steps apply unchanged.

**See also:** [kht-detect](kht-detect.md), [adrt-api](adrt-api.md),
[testdata](testdata.md), [geom-line](geom-line.md), `../docs/LFD_DESIGN.md`
