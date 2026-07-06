# LFD_DESIGN.md — ADRT-based Linear Feature Detector

Design report and step-by-step implementation recommendation for a Linear
Feature Detector (LFD) built on the **Approximate Discrete Radon Transform
(ADRT)**, for astronomical difference images.

- **Status:** design proposal (2026-07-02). Evolving — revise as experiments settle.
- **Grounding:** `adrt` v1.2.0 API/geometry verified locally (see
  `knowledge/adrt-api.md`); KHT/Hough facts from the references at the end.
- **Scope:** detection of linear features (satellite/aircraft streaks, some
  cosmic-ray tracks, diffraction spikes) as line parameterizations. Notebooks in
  this repo are scratch and are **not** treated as a spec.

---

## 1. Reference implementation (Hough/KHT LFD)

The existing astronomical LFD we mirror:

- **Inputs:** difference image, bad-pixel mask, variance plane.
- **Steps:**
  1. Threshold → binary image (`True` = pixel above threshold).
  2. Canny edge transform.
  3. Mask bad pixels.
  4. Kernel Hough Transform (KHT).
  5. Cluster pairs of lines corresponding to the two parallel edges of a feature.
- **Output:** list of lines in **Hesse normal form** `ρ = x·cosθ + y·sinθ`.

The ADRT implementation follows the same shape: prepare → transform → detect
peaks → post-process → emit Hesse-form lines.

---

## 2. Why ADRT instead of KHT — and the algorithmic differences

Both KHT and the ADRT find lines by mapping image content into a `(ρ, θ)`
parameter space and locating concentrations. They differ in **what votes**,
**how the accumulator is built**, and therefore in what pre-processing and
post-processing are appropriate.

| Aspect | Kernel Hough Transform (KHT) | ADRT (this design) |
|---|---|---|
| Primitive that votes | **Edge pixels** (post-Canny), clustered into approximately-collinear strings; each cluster votes with an oriented elliptical-Gaussian kernel | **Pixel intensities** summed along digital lines — the whole image is transformed, no edge detection required |
| Input | Binary/edge map | Real-valued image (ideally intensity, weighted by significance) |
| Accumulator | Sparse, kernel-smoothed votes in continuous `(ρ,θ)` | Dense array, exact partial sums along `O(N)` discrete line families |
| Cost | `O(edges)` + clustering; fast on sparse edges | `O(N² log N)` for an `N×N` image; independent of feature count |
| Angle/offset sampling | Continuous, chosen adaptively | Fixed digital lines; `θ∈[−90°,+90°]` in four 45° quadrants, `2N−1` offsets |
| Noise model | Thresholded away before voting | **Carried through** — can weight sums by variance |
| Line "thickness" | Two edges of a streak ⇒ two Hough peaks (hence the pairing step) | A bright streak is one ridge; both edges only appear if you edge-filter first |

### Key algorithmic modifications relative to KHT

1. **Drop Canny; transform intensities directly (biggest change).**
   The ADRT integrates signal along lines, so it accumulates faint,
   sub-threshold flux coherently — the classic strength of Radon methods over
   Hough for low-SNR streaks. Do **not** feed it a binary edge map by default;
   feed a **variance-weighted, sky-subtracted image**. This changes the front of
   the pipeline substantially (Steps 1–3 below).

2. **Replace the "pair of parallel edges" clustering with single-ridge peak
   detection.**
   Because we transform the streak body (not its two edges), a streak is **one**
   peak in ADRT space, not a pair. The KHT pairing step (Step 5) is *removed*;
   width is recovered instead from the peak's offset-extent or a profile fit
   across the ridge. (If an edge-based variant is ever needed, the pairing logic
   returns — see §6.)

3. **Quadrant-aware peak finding.**
   ADRT output is four quadrants `(4, 2N−1, N)`, each a distinct 45° angle band.
   Detect peaks per quadrant (or on the `stitch_adrt` mosaic with care at the
   seams), because a single global 2-D peak finder will mishandle the quadrant
   boundaries and the differing offset scaling.

4. **Matched-filter / multiscale detection instead of a smoothing kernel.**
   KHT's elliptical-Gaussian kernel models line uncertainty at vote time. The
   ADRT analogue is to **matched-filter the accumulator** along the offset axis
   with the expected streak cross-section (a short 1-D kernel ≈ PSF width), and
   optionally to run the ADRT at multiple padded scales. This localizes ridges
   and suppresses spurious peaks — the same goal as KHT's cleaner accumulator,
   achieved in accumulator space rather than at vote time.

5. **Coordinate conversion via `coord_adrt`, not a Hough `(ρ,θ)` grid.**
   ADRT indices are not directly `(ρ,θ)`. Use `adrt.utils.coord_adrt(N)` to map
   each accumulator cell to physical `(offset, angle)`, then convert to the
   image's Hesse normal form. This is the ADRT counterpart of reading `(ρ,θ)`
   off the Hough accumulator.

6. **Exploit the exact inverse for validation (no Hough equivalent).**
   `adrt.iadrt` / `iadrt_fmg` / `bdrt` let us back-project a detected peak into
   image space to confirm a real streak and estimate its extent/endpoints — a
   verification step KHT cannot offer cheaply.

---

## 3. Inputs and preconditions

Same inputs as the reference LFD:

- **Difference image** `D` — float, sky/background ≈ 0.
- **Bad-pixel mask** `M` — `True` = bad/ignore.
- **Variance plane** `V` — per-pixel variance (> 0 where valid).

ADRT preconditions (verified, `adrt` v1.2.0):

- Input must be **square**, side `N` a **power of two**, float dtype.
- Output shape `(4, 2N−1, N)`: axis0 = 4 quadrants, axis1 = offset (`2N−1`),
  axis2 = angle (`N`). Angles span `[−90°, +90°]` — quadrants cover
  `[−90,−45], [−45,0], [0,45], [45,90]` degrees.

---

## 4. Recommended step-by-step pipeline

Mirrors the reference (prepare → transform → detect → post-process → emit),
with the ADRT modifications from §2.

### Step 0 — Prepare & normalize the image (new)
- Compute a **significance / weighted image**: `S = D / sqrt(V)` (per-pixel S/N),
  or a variance-weighted `D`. This is what gets transformed — it lets the ADRT
  sum in units where noise is homogeneous, so a line integral's significance
  grows with length.
- Zero-out or interpolate over bad pixels using `M` (see Step 2).
- **Pad to a power-of-two square** with `numpy.pad` (constant 0), recording the
  pad offsets so coordinates can be mapped back to the original frame.
  - *Astronomy note:* detector images (e.g. 4096-wide LSST amps/CCDs) are rarely
    power-of-two squares; padding policy (per-amp vs. per-CCD vs. per-detector,
    and how to handle non-square) is a first-class design decision. Record it in
    `knowledge/`.

### Step 1 — Build the detection input (replaces "binary threshold")
- **Default (recommended): keep it real-valued** — do *not* binarize. Optionally
  soft-threshold: `S_det = max(S − k, 0)` for a low `k` (e.g. 1–2σ) to suppress
  pure noise while preserving faint coherent flux for the line integral.
- Provide a **binary mode** (`S_det = S > k`) for parity with the Hough baseline
  and for A/B comparison, but it is not the primary path.

### Step 2 — Handle bad pixels (moved earlier; replaces post-Canny masking)
- Bad pixels must be neutralized **before** the transform, because the ADRT sums
  along lines and a bad pixel contaminates every line through it.
- Set masked pixels to **0** in the weighted image (0 contributes nothing to a
  sum). For strong artifacts, interpolate across `M` first.
- **Track masked fraction per line:** transform the mask (or `1−M`) *alongside*
  the image so each accumulator cell knows how many valid pixels it integrated;
  use this to normalize/penalize lines that cross many bad pixels. This is the
  ADRT replacement for edge-map masking.

### Step 3 — Forward ADRT (replaces Canny + KHT voting)
- `A = adrt.adrt(S_det)` → `(4, 2N−1, N)`.
- Optionally transform the valid-pixel mask: `A_count = adrt.adrt((~M).astype(float))`
  for per-line normalization (Step 4).

### Step 4 — Accumulator conditioning & peak detection (replaces KHT accumulator)
- **Normalize** for line length / valid-pixel count using `A_count` so short or
  heavily-masked lines don't masquerade as strong detections.
- **Matched-filter** along the offset axis with a small kernel matched to the
  streak cross-section (~PSF FWHM) — the ADRT analogue of KHT's Gaussian kernel.
- **Detect peaks per quadrant:** local maxima above an adaptive threshold
  (e.g. `median + k·MAD` of the accumulator). Enforce a minimum separation in
  `(offset, angle)` to avoid duplicate detections of one ridge.
- Optionally visualize with `adrt.utils.stitch_adrt` during development.

### Step 5 — Convert peaks to line parameters (replaces edge-pair clustering)
- `offset, angle = adrt.utils.coord_adrt(N)`; index the peak cells to read
  physical `(offset, angle)`.
- Convert to **Hesse normal form** `ρ = x·cosθ + y·sinθ` **in original image
  coordinates**, undoing the Step 0 padding/shift.
- Recover width from the ridge's offset-extent (or a cross-ridge profile fit);
  recover endpoints via back-projection (Step 6). *No parallel-edge pairing is
  needed* — one streak → one line.

### Step 6 — Verification & refinement (new; uses the inverse)
- Back-project each candidate (`iadrt`/`bdrt`, or synthesize the line) and check
  it against the difference image to reject spurious peaks and estimate
  endpoints/length.
- Astronomy-specific vetoes: reject lines aligned with detector columns/rows or
  bleed trails; cross-check against the mask; optionally require a plausible
  streak profile.

### Output
- List of detections, each with: **Hesse `(ρ, θ)`** in image coordinates,
  plus significance, estimated width, endpoints, and valid-pixel fraction.

---

## 5. Astronomy-specific modifications (summary)

1. **Variance-weighted input** (`D/√V`) so detection significance is calibrated
   and faint long streaks integrate coherently.
2. **Mask propagated through the transform** (transform `1−M` for per-line valid
   counts) rather than masking an edge map.
3. **Padding/tiling policy** for non-power-of-two, non-square detector data,
   with coordinate bookkeeping back to sky/pixel frame.
4. **PSF-matched offset filtering** in the accumulator.
5. **Physical vetoes** (detector-aligned artifacts, bleed trails, cosmic rays)
   and optional WCS conversion of `(ρ,θ)` to on-sky orientation.
6. **Back-projection verification** to control false positives — critical at the
   survey scale where even low false-positive rates flood downstream catalogs.

---

## 6. Open questions / decisions to record in `knowledge/`

- Tiling vs. single-pad for full detectors; overlap handling for streaks
  crossing tile boundaries.
- Threshold strategy (soft-threshold `k`, accumulator `k·MAD`) and how to make
  it survey/depth-adaptive.
- Whether an **edge-based ADRT variant** (Canny → ADRT → *reinstate* the
  parallel-edge pairing of Step 5) is worth keeping for wide/saturated streaks.
- Handling curved or broken features (multiple short segments vs. one line).
- Benchmark plan: ADRT-LFD vs. the KHT baseline on labeled difference images
  (completeness/purity vs. streak SNR, width, length).

---

## References

- ADRT package (Otness & Rim, JOSS 2023) and its algorithm sources
  (Brady 1998; Press 2006) — see `adrt` docs `cite` page and
  `knowledge/adrt-api.md` for the locally verified API/geometry.
- Kernel-based Hough Transform: Fernandes & Oliveira, *"Real-time line detection
  through an improved Hough transform voting scheme,"* Pattern Recognition, 2008
  (clusters approximately-collinear edge pixels; votes with an oriented
  elliptical-Gaussian kernel for a cleaner accumulator).
- Hough transform / Hesse normal form `ρ = x·cosθ + y·sinθ`.

> Note: intended web searches for the KHT and ADRT source papers could not be
> run (WebSearch unavailable for this model; several `adrt` doc/example URLs
> returned 404). The comparison above rests on the locally verified `adrt` v1.2.0
> API plus standard KHT/Hough descriptions; confirm the paper citations before
> publication.
