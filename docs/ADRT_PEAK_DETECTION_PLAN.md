# ADRT_PEAK_DETECTION_PLAN.md — Peak detection for the ADRT LFD

Design plan for the **peak-detection stage** of the ADRT linear-feature detector
(`docs/LFD_DESIGN.md` §4 Step 4–5). This is the stage that turns the raw ADRT
accumulator into a small set of candidate line parameters, and it must feed a
downstream **butterfly analysis** (Hough-transform-style neighborhood analysis of
the accumulator around each peak) that estimates feature properties.

- **Status:** design proposal (2026-07-07). Evolving — revise as experiments settle.
- **Grounding:** `knowledge/adrt-api.md`, `knowledge/detector-task.md`,
  `knowledge/testdata.md`, `knowledge/geom-line.md`; `adrt` v1.2.0 verified locally.
- **Scope:** peak detection *only* — from `A = adrt.adrt(S_det)` to a list of
  candidate peaks + neighborhood accessors. Coordinate conversion to Hesse form,
  the profile fit, and output-catalog assembly are shared/downstream and are
  described elsewhere (`knowledge/detector-task.md`).

---

## 1. Context and confirmed requirements

The peak detector operates on the **unstitched** ADRT output — the four
accumulator arrays in slope–height coordinates:

```
A = adrt.adrt(S_det)          # shape (4, 2N-1, N), float
#   axis 0 : quadrant   (4 angle bands, each spanning 45°)
#   axis 1 : offset     (2N-1)  — the "height"/intercept
#   axis 2 : angle      (N)     — the "slope"
```

Each accumulator cell is the sum of the significance image `S_det = D/√V` (see
`LFD_DESIGN.md` Step 0–1) along one digital line. A **line in image space maps to
one localized peak** in this space.

Design decisions confirmed with the user (2026-07-07):

1. **Few features per image, but ridges may be *fragmented*.** A feature broken
   into several collinear segments still deposits into the **same (offset, angle)
   ridge** → it remains **one peak**, not several. Fragmentation shows up as a
   ridge whose along-line flux is only partially filled, *not* as extra peaks.
   ⇒ Peak *detection* stays in the simple "few, well-separated peaks" regime;
   segment/length recovery is a separate concern.
2. **Detector flags partial ridges.** Alongside each peak, report a coarse
   *along-line occupancy* hint derived from the valid-pixel count transform
   (`A_count`, `LFD_DESIGN.md` Step 3), so downstream knows a ridge is fragmented.
3. **Coords + helper for neighborhoods.** The detector returns peak
   coordinates/values; a **reusable `extract_window()` helper** cuts the
   surrounding accumulator patch (the "butterfly") on demand, handling quadrant
   edges and offset padding. The detector does not force a patch into every result.
4. **Refinement is decoupled.** The detector returns **integer-grid** peaks. All
   sub-pixel refinement lives in the downstream butterfly analysis. The detector
   exposes the accessors that refinement needs (window helper + coord mapping) but
   does not itself iterate.

These four decisions shape every option below.

---

## 2. Why ADRT peak-finding is not generic 2-D peak-finding

Pitfalls specific to this accumulator (record in `knowledge/adrt-geometry.md`
when settled):

- **Quadrant seams (axis 0).** The four quadrants are *separate* 45° angle bands
  with independent geometry. A global 2-D maximum filter across a stitched mosaic
  will (a) find false maxima at seams and (b) mishandle the fact that offset
  scaling differs across the band. **Detect per quadrant** on the unstitched
  arrays — this is also exactly the layout the butterfly analysis wants.
- **Offset axis is `2N-1`, not `N`** — do not transpose. The angle axis is `N`.
- **Ridge shape is anisotropic.** A true line is sharp across offset (≈ streak
  width ⊛ PSF) and broad along angle — the classic Hough "butterfly". Peak
  criteria and windows should be **anisotropic** (wider along angle than offset).
- **Length/valid-pixel bias.** Longer digital lines and lines crossing fewer bad
  pixels accumulate more — raw peak height is *not* significance. Condition the
  accumulator with `A_count` (Step 4 of the design) before thresholding.
- **DC / low-frequency pedestal.** Even a sky-subtracted `S_det` leaves a smooth
  background in `A`; subtract a local baseline (per-quadrant median or a large
  median filter) before peak-finding so the threshold is meaningful.

---

## 3. Proposed interface

A single small module `src/astro_lfd/meas/adrtPeaks.py` (stack-independent —
depends only on `numpy`/`scipy`/`adrt`, mirroring how `utils/` stays stack-free;
the LSST bridge happens later in the detector task). Core dataclass:

```python
@dataclass(frozen=True)
class AdrtPeak:
    quadrant: int          # 0..3
    offset_idx: int        # index on the 2N-1 axis
    angle_idx: int         # index on the N axis
    value: float           # conditioned accumulator value at the peak
    offset: float          # physical offset  (from coord_adrt)
    angle: float           # physical angle [rad] (from coord_adrt)
    occupancy: float | None  # along-line valid/filled fraction hint (Step 2 below)
    significance: float | None  # value / noise scale (e.g. MAD), if computed
```

Primary functions:

```python
def detect_peaks(A, *, A_count=None, config=PeakConfig()) -> list[AdrtPeak]: ...

def extract_window(A, peak, *, half=(dh_offset, dh_angle),
                   quadrant_only=True) -> Window: ...   # the butterfly patch
```

`Window` carries the sliced sub-array **plus** the physical `(offset, angle)`
coordinates of every cell in it (sliced from `coord_adrt(N)`), so the downstream
butterfly analysis can do sub-pixel centroiding/quadratic fits *without* re-deriving
geometry. Returning coordinates-with-patch is the single most useful thing for the
"ease of extraction" and "refinement" requirements.

`coord_adrt(N)` is called **once** and cached (offset is `(4,2N-1,N)`, angle is
`(4,1,N)` — broadcastable); both `detect_peaks` and `extract_window` slice from it.

---

## 4. Pipeline and options

The stage decomposes into four sub-steps; each offers speed/accuracy/sensitivity
options. **Bold = recommended default.**

### Step A — Accumulator conditioning (before any peak-finding)

Turn raw sums into a comparable-across-cells detection statistic.

- **A0 (default): length/count normalization + baseline subtraction.**
  Normalize by valid-pixel count from `A_count = adrt.adrt((~M).astype(float))`
  (guard divide-by-zero at short lines), then subtract a per-quadrant baseline
  (median or large-kernel median filter). Cheap, robust.
- A1 (higher sensitivity): **PSF-matched filter along the offset axis** with a
  short 1-D kernel ≈ streak-cross-section (PSF FWHM). Sharpens ridges, suppresses
  single-pixel noise spikes. Adds one `scipy.ndimage.correlate1d` per quadrant.
- A2 (fastest / baseline parity): none — threshold raw `A`. Only for A/B tests;
  biased toward long lines.

Noise scale for thresholding: **per-quadrant `median + k·MAD`** (robust to the
few real ridges). `k` in config (default ~5). MAD is cheap and streak-count-robust.

### Step B — Candidate localization (per quadrant)

- **B0 (default): local-maxima via `scipy.ndimage.maximum_filter`** with an
  **anisotropic footprint** (larger along angle than offset), keep cells equal to
  the filtered max and above threshold. Enforce **minimum separation** in
  `(offset, angle)` (non-max suppression) so one broad ridge yields one peak.
  O(N² log N)-free, vectorized, handles the "few peaks" case directly.
- B1 (crowded / robustness): **iterative detect-and-suppress (CLEAN-like).** Take
  the current global max in the quadrant, record it, zero a small region around it
  (or subtract a modeled ridge), repeat until below threshold or `max_peaks` hit.
  Not needed for the expected few-feature regime, but the cleanest way to stop a
  bright ridge from masking a faint neighbor. Keep behind a config flag.
- B2 (label-based): threshold → `scipy.ndimage.label` connected components →
  take the max cell per component. Naturally groups a fat ridge into one peak and
  gives the component extent for free (useful as a window prior). Slightly more
  bookkeeping; good when ridges are broad.

All three run **per quadrant** on the unstitched arrays — never on the stitched
mosaic (§2).

### Step C — Partial-ridge occupancy hint (the "flag partial ridges" requirement)

A true full-length line has `A_count` at the peak cell ≈ the full digital-line
length; a fragmented feature fills only part of it. Two cheap ways to summarize:

- **C0 (default): count-ratio.** `occupancy = A_count[peak] / L_full(peak)`,
  where `L_full` is the nominal digital-line length for that cell (derivable from
  the geometry / obtainable by transforming an all-ones image once). Near 1 ⇒
  full line; well below ⇒ fragmented or heavily masked. One extra array lookup.
- C1 (richer): also transform a **binary detection mask** `A_det = adrt.adrt(S_det>k)`
  and compare `A_det[peak]` to `L_full` — distinguishes "line present but faint"
  from "line present over a short span". Extra ADRT call; optional.
- Note: distinguishing *fragmentation* from *masking* needs both `A_count`
  (valid pixels) and `A_det` (filled-with-signal pixels). Expose both hints; leave
  the interpretation to the downstream/segment-recovery step. **Full endpoint /
  segment recovery is explicitly downstream** (back-projection, `LFD_DESIGN.md`
  Step 6) — the detector only *flags*.

### Step D — Physical coordinates + neighborhood accessor

- Map each peak `(quadrant, offset_idx, angle_idx)` → physical `(offset, angle)`
  by indexing the cached `coord_adrt(N)` arrays. (Conversion of `(offset, angle)`
  → image Hesse `(rho, theta)` and un-padding is the **next** pipeline stage,
  shared with the detector task — out of scope here, but the peak carries what it
  needs.)
- `extract_window(A, peak, half=...)` returns the butterfly patch **clamped to the
  quadrant** (default) plus its per-cell `(offset, angle)` coordinates. Clamping to
  the quadrant avoids the seam problem; an optional `quadrant_only=False` mode can
  stitch across a seam later if a peak sits at a band edge (deferred).

---

## 5. How this serves the downstream butterfly analysis

The butterfly analysis inspects accumulator values *around* a peak (as in Hough
"butterfly" spread analysis) to estimate line properties (angle uncertainty,
width, and — via sub-pixel centroiding — a refined location). This plan supports
it deliberately:

- **Ease of extraction:** `extract_window()` hands back a ready-to-analyze patch
  *with coordinates attached*, so the downstream code never re-indexes the
  `(4, 2N-1, N)` layout or re-derives `coord_adrt`. Quadrant/edge handling is
  solved once, in the helper.
- **Refinement (decoupled):** the detector returns integer-grid peaks; the
  butterfly analysis computes any sub-pixel `(offset, angle)` from the window
  (e.g. a 2-D quadratic fit or intensity centroid) and owns the result. Because
  the window carries physical coordinates, the refined value is directly in
  `(offset, angle)` — no dependence on detector internals. If a re-extract is ever
  wanted after refinement, the caller just calls `extract_window()` again around
  the refined cell; no feedback API is baked into the detector.
- **Anisotropy:** default window `half` is larger along angle than offset,
  matching the butterfly's shape, so the patch actually contains the wings the
  analysis needs.

---

## 6. Test plan

Split per the task into (a) tests runnable with **existing tools** and (b) tests
needing **new code**. Follow repo conventions: `pytest` in `tests/`, `black`,
type hints; ADRT-ready inputs via `testdata.simulate_exposure(shape=(256,256))`
(pow2 square, no padding — `knowledge/testdata.md`).

### 6a. Tests using existing tools (no new production code)

These validate assumptions and calibrate defaults; they can be written now against
`adrt` + `astro_lfd.utils.testdata` before `adrtPeaks.py` exists.

1. **Ground-truth peak location.** Simulate a single streak of known `(rho, theta)`
   (`StreakConfig`), build `S_det`, `A = adrt.adrt(S_det)`. Assert the global-max
   cell, mapped through `coord_adrt`, corresponds to the injected line within
   tolerance. *Locks the offset/angle-axis and quadrant conventions before trusting
   any detector.*
2. **One line → one peak (fragmentation invariance).** Inject a line, then the same
   line broken into 2–3 collinear segments (gaps). Assert the peak `(offset, angle)`
   cell is unchanged and there is still a single dominant peak; assert `A_count`- or
   `A_det`-based occupancy at that cell *drops* for the fragmented case. *Directly
   validates the core assumption behind the "few peaks, flag partial ridges" design.*
3. **Length/count bias.** Compare raw `A` peak height vs. count-normalized value for
   a long vs. short line at equal per-pixel SNR; confirm normalization equalizes
   them. *Calibrates Step A0.*
4. **Baseline/threshold behavior.** On a noise-only `S_det`, confirm `median+k·MAD`
   yields ~zero detections at the chosen `k` (false-positive rate calibration).
5. **Quadrant-seam sanity.** Confirm lines placed to fall near the ±45° band
   boundaries land in the expected quadrant and that a naive stitched-mosaic max
   would mislocate them (documents *why* per-quadrant detection is used).

### 6b. Tests requiring new code (the peak-detection module)

Written against `src/astro_lfd/meas/adrtPeaks.py`:

6. **`detect_peaks` — recovery & multiplicity.** N injected non-overlapping lines
   ⇒ exactly N peaks at the right cells; robustness to added Gaussian noise
   (completeness vs. SNR curve, a few seeds). Parametrized over Step-B options
   (B0/B1/B2) to compare sensitivity.
7. **Minimum-separation / non-max suppression.** A single broad ridge must yield
   **one** peak, not a cluster (guards the "fragmented ⇒ still one peak" contract at
   the detector level).
8. **`extract_window` correctness.** Window sub-array equals the manual slice of
   `A`; attached coordinates equal the matching slice of `coord_adrt(N)`; clamping
   at quadrant/offset edges returns a valid (possibly truncated) patch with no
   out-of-bounds indexing. *This is the interface the butterfly analysis relies on.*
9. **Occupancy hint.** For full vs. fragmented vs. mask-crossing lines, assert the
   reported `occupancy` orders as expected (full ≈ 1 > fragmented; mask-crossing
   reduced). Covers Step C.
10. **Refinement-readiness (decoupled).** A stub sub-pixel centroid on the window
    recovers a known sub-cell offset to < 1 cell — confirms the window+coords
    payload is sufficient for downstream refinement without any detector feedback API.
11. **Regression/perf smoke.** `detect_peaks` on a `(4, 511, 256)` accumulator runs
    within a time budget and returns stable results across runs (vectorized paths,
    no accidental O(N³)). Optional benchmark comparing A0 vs. A1 (matched filter)
    cost.

### 6c. Optimization / validation experiments (notebook-level, not unit tests)

- Sweep `k` (MAD threshold), window `half`, and matched-filter width on a small
  labeled set (vary streak SNR/width/length) → completeness/purity curves. Graduate
  chosen defaults into `PeakConfig`.
- Cross-check a handful of detected peaks by **back-projection** (`iadrt`/`bdrt`,
  `LFD_DESIGN.md` Step 6) to confirm they reconstruct the injected line — a
  detector-independent validation of the whole chain.

---

## 7. Implementation specifics (deferred to coding)

Notes to pick up when actually writing `adrtPeaks.py` — *not* decisions to make now:

- Cache `coord_adrt(N)` per `N` (module-level `functools.lru_cache`); slice, don't
  broadcast, to keep memory down (angle is `(4,1,N)`).
- Vectorize per-quadrant local maxima with `scipy.ndimage.maximum_filter`
  (anisotropic `footprint`), then boolean-mask against the threshold; apply NMS by
  sorting candidates by value and greedily dropping neighbors within `min_sep`.
- `PeakConfig` fields (snake_case, per repo convention): `mad_k`, `min_sep_offset`,
  `min_sep_angle`, `matched_filter_fwhm | None`, `max_peaks`, `window_half_offset`,
  `window_half_angle`, `detect_mode ∈ {local_max, clean, label}`, `use_count_norm`.
- `L_full(cell)` for occupancy: obtain once via `adrt.adrt(np.ones((N,N)))` (nominal
  per-cell line length) rather than a closed form — robust to any geometry subtlety.
- Add shape/dtype assertions at the boundary (`A.shape == (4, 2N-1, N)`, float) per
  CLAUDE.md's "assert at boundaries" rule; assert `A_count`/`A` shapes match.
- Keep the module stack-independent; the detector task (`meas/detectStreaks.py`
  sibling) imports it and bridges to `lsst.afw` — the peak module itself imports no
  `lsst.*`.
- Conventions/gotchas are recorded in `knowledge/adrt-geometry.md` (created with
  this plan); refine it and add a dedicated peak-detection note as the module lands.

---

## 8. Open questions to resolve during implementation

- Matched-filter default on or off? (A1 improves faint-streak sensitivity but adds
  a tunable width — decide from the 6c SNR sweep.)
- Do we need `A_det` (binary-transform) occupancy in addition to `A_count`, or is
  the count-ratio enough to flag fragmentation for the first version?
- Cross-seam windows: needed for real data, or is quadrant-clamped extraction
  sufficient until a peak is observed to straddle a boundary?
- Threshold adaptivity across survey depth (global `k·MAD` vs. spatially-varying) —
  likely deferred until real diffim runs.

---

## References

- `docs/LFD_DESIGN.md` — full ADRT-LFD pipeline (this plan details Step 4–5).
- `knowledge/adrt-api.md` — `adrt` v1.2.0 shapes/signatures (`coord_adrt`, layout).
- `knowledge/detector-task.md`, `knowledge/kht-detect.md` — shared detector-task
  formulation, output format, and the KHT baseline this will be compared against.
- `knowledge/testdata.md` — synthetic ADRT-ready inputs for the tests above.
</content>
</invoke>
