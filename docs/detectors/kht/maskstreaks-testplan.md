# Test plan: KHTDetectTask vs. MaskStreaksTask output comparison

**Status:** awaiting review — inputs/fixtures must be provided before execution.

## Goal

Empirically compare the streak lines produced by the new
`astro_lfd.meas.detectStreaks.KHTDetectTask` against the `lines` produced by
`lsst.meas.algorithms.maskStreaks.MaskStreaksTask`, on the **same** input image,
to confirm that the reformulation preserves detection behavior and to quantify
any residual discrepancy. The ranked, code-derived hypotheses for *why* they can
differ are in `maskstreaks-discrepancy.md`; this plan is how we would
measure them.

**Constraint (from the task):** do **not** run a full pipeline / `PipelineTask`.
The harness below calls the two `Task.run`/`find` methods directly on an
in-memory exposure. Anything requiring a Butler, a data repository, or real
survey data is out of scope until reviewed.

## What the two tasks emit (so we know what to compare)

| | `MaskStreaksTask` | `KHTDetectTask` |
|---|---|---|
| Entry point | `find(maskedImage)` / `run(maskedImage)` | `run(table, exposure)` |
| Line output | `Struct.lines` — `LineCollection` of `Line(rho, theta, sigma, reducedChi2, modelMaximum)` | `Struct.streaks` — `SourceCatalog` of `StreakAdapter` rows |
| Frame | image-**center**-relative `rho`, `theta` in **degrees** (see discrepancy §1) | absolute pixel frame; `Line2D` canonicalized to `theta∈[0,π)` |
| Also | `mask` (union), `lineClusters`, `originalLines` | `edges`, per-line `Footprint`, sky `coord` |

The comparison must therefore **map both to one common frame** before matching
(see "Canonicalization" below).

## Inputs required (to be supplied / confirmed by reviewer)

Both tasks consume an `lsst.afw.image` masked image / exposure with:

1. A **`DETECTED`** mask plane flagging the streak pixels (both use it as the
   Canny seed). Without it, both return zero lines.
2. **Bad planes** (`NO_DATA`, `INTRP`, `BAD`, `SAT`, `EDGE`, and — for
   `KHTDetectTask` only — `ITL_DIP`, `SPIKE`) present in the mask dictionary.
3. Finite `image` and `variance` arrays (the profile fit uses `1/variance`).
4. For `KHTDetectTask` only: optionally a **WCS** (for `setCoord`) and a schema
   built by `StreakAdapter.makeMinimalSchema()`. The WCS is guarded (skipped if
   `None`), so it is not required for the line geometry itself.

**Open question for review — which input source?** Options, in increasing realism:

- **(A) Extend `astro_lfd.utils.testdata`** to also emit a `DETECTED` plane
  (threshold the noise-free `streak_signal`) and, optionally, a trivial WCS.
  Pros: no external data, fully deterministic geometry, already produces
  image/variance/mask in the right shapes and an `ExposureF` via `to_exposure`.
  Cons: single clean streak, Gaussian-ish edges — exercises the happy path only.
  **Recommended starting point** — needs a small, reviewable testdata addition.
- **(B) A real calexp / difference image** with `DETECTED` and `SAT` already set,
  containing a known satellite streak. Pros: exercises the SAT-dilation and
  bad-plane paths that (A) cannot. Cons: needs a data file + provenance from the
  reviewer; larger, non-deterministic.

Please confirm (A), (B), or both — and for (B) provide the file path / dataId and
which mask planes are populated.

## Harness (sketch — not to be run until inputs are approved)

```python
# Pseudocode. Runs the two detection methods directly (no PipelineTask).
exposure = <approved input>            # ExposureF with DETECTED etc. set
mi = exposure.maskedImage

# --- MaskStreaksTask (reference) ---
ms = MaskStreaksTask()
ms_out = ms.find(mi)                    # .lines : LineCollection

# --- KHTDetectTask (new) ---
schema = StreakAdapter.makeMinimalSchema()
table = afwTable.SourceTable.make(schema)
kht = KHTDetectTask()
kht_out = kht.run(table, exposure)      # .streaks : SourceCatalog

# --- Canonicalize both to one frame, then match ---
ref  = [to_common(l) for l in ms_out.lines]          # see below
test = [to_common(StreakAdapter(r).getLine()) for r in kht_out.streaks]
pairs = match_by_rho_theta(ref, test, rho_tol=rhoBinSize, theta_tol=thetaBinSize)
report_deltas(pairs)                     # d_rho, d_theta, unmatched on each side
```

### Canonicalization (the crux)

To compare fairly, convert **both** outputs into the same representation:

1. Put both into the **same origin**. `MaskStreaksTask` lines are relative to the
   image center (`rho = x·cosθ + y·sinθ` with `x,y ∈ [-N/2, N/2)`);
   `KHTDetectTask` already translates its `Line2D` by the bbox center into
   absolute pixels. Either shift the reference lines by `+bboxCenter` or shift the
   test lines back by `-bboxCenter` — pick one and apply consistently. Watch for
   the `XY0` offset if the exposure bbox does not start at `(0, 0)`.
2. Apply the **same angle canonicalization** to both (fold `theta` into `[0, π)`,
   flipping `rho`'s sign when subtracting π — this is exactly what
   `Line2D._canonicalize` does). Convert `MaskStreaksTask`'s degrees accordingly.
3. Match on `(rho, theta)` within tolerances tied to the clustering bin sizes
   (`rho_bin_size`, `theta_bin_size`). `astro_lfd.geom.line.embed_rho_theta` +
   a nearest-neighbor pairing is a convenient, tolerance-scaled matcher.

## Metrics to report

- **Count agreement:** number of lines from each task; matched / unmatched.
- **Per-match residuals:** `|Δrho|` (pixels), `|Δtheta|` (deg), and `Δlength`
  (KHTDetectTask segment length vs. the reference line clipped to the same bbox).
- **Unmatched lines** on each side, with which discrepancy hypothesis
  (`maskstreaks-discrepancy.md`) most plausibly explains each.
- **Determinism control:** run the KMeans-based clustering ≥3× (fixed input) to
  separate genuine code differences from KMeans random-init jitter (§4 there).

## Controls / knobs to isolate causes

- Set both configs to **identical** shared values (bin sizes, kernel heights,
  `invSigma`, tolerances) so config drift is not conflated with code drift.
- Temporarily align `bad_mask_planes` (drop `ITL_DIP`, `SPIKE`) to test whether
  the extended bad-plane set (§3) is the driver.
- Seed / fix KMeans (`n_init`, `random_state`) if feasible to remove §4 jitter.

## Deliverable

A short results table (counts, residual distributions) appended to
`maskstreaks-discrepancy.md`, confirming or re-ranking the hypotheses there.

---
**Next action:** reviewer confirms the input source (A/B) and provides any real
data file + populated mask planes. Execution is paused until then.
