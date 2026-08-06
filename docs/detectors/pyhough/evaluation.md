# pyhough — evaluation as a candidate LFD detector

Assessment of the legacy **`pyhough`** package (E. Rykoff, SLAC; GPL) as a
potential line-finding core within `astro_lfd`. The intent is to place `pyhough`
in context — first among linear-feature detectors generally, then directly
against the current reference detector [KHT](../kht/) — and only then to derive
recommended changes for future testing.

- **Status:** evaluation (2026-07-31). Design/architecture review only.
- **Grounding:** `pyhough` and `lsst.kht` source read in full, not executed.
  - `pyhough`: `pyhough/pyhough_lib.py`, `pyhough/pyhough.c`,
    `pyhough/pyhough_pywrap.c`, `pyhough/pyhoughback.c`, and the demonstration
    driver `simple_streak_finder.py` (`~/rubin-user/git-owners/pyhough/`).
  - `kht`: `cpp/source/{linking,subdivision,voting,peak_detection}.cpp`,
    `cpp/include/kht/kht.hpp`, `cpp/lsst/kht/{kht.cc,khtContinued.py}`
    (`~/rubin-user/lsst/kht/`), plus [`knowledge/kht-detect.md`](../../../knowledge/kht-detect.md).
- **Scope note:** `pyhough` is judged on its own merits as an architecture, not
  as a critique of an older codebase. Reports §1 and §2 capture *reasoning*;
  §3 turns that reasoning into directive changes but **does not implement them**.

---

## What `pyhough` is (baseline facts)

`pyhough.Hough(image).transform()` is a **classical Standard Hough Transform
(SHT)** with a thin C core:

- **Input:** a 2-D **binary** array (`image.astype(bool)`). No edge detection,
  linking, or intensity handling is built in — every `True` pixel is a voter.
- **Voting** (`_hough_transform`, `pyhough.c`): for each of `ntheta` angles it
  loops over all set pixels and increments one accumulator cell per pixel via
  `index = (long)(x·cosθ + y·sinθ + offset + 0.5)` ("poor man's rounding" — hard
  nearest-`rho` binning, valid because `offset` forces positivity). One hard
  `+1` per (pixel, θ). No smoothing, no interpolation, no uncertainty.
- **Parameter space:** `rho = x·cosθ + y·sinθ` with the origin at the
  **lower-left corner**; `θ ∈ [0, π)` in radians; `drho = 1` (hardcoded);
  `ntheta`, `nrho` auto-sized to the image diagonal (`nrho = 2·⌈√(r²+c²)⌉+1`).
- **Accumulator dtype:** `uint16` (`NPY_UINT16` in `pyhough_pywrap.c`).
- **Output:** `(accumulator, theta, rho)` — the raw vote map plus its axes.
  **Peak detection is not part of the library**; the caller owns it. A
  companion back-projection module (`pyhoughback.c`, `Back`) exists but is unused
  by the driver.

The **`simple_streak_finder.py`** driver supplies the astronomy pipeline around
that core:

1. Bin the exposure `16×` (`afwMath.binImage`) — SNR gain + smaller accumulator.
2. Estimate sky noise (`summaryStats.skyNoise`, else MAD of unmasked pixels).
3. Threshold at `nsig_det·(noise/binning)` → binary `det`; zero bad-masked pixels.
4. Hough-transform `det` (numerator) **and** an all-ones "template" with the same
   bad pixels removed (denominator).
5. `ratio = transform / transform_template` — normalizes each cell's votes by the
   number of **valid pixels a line could cross**.
6. Threshold `ratio` at `median + nsig_streak·MAD` **and** require the template
   count to exceed `min_streak_length/binning` (long-enough lines only).
7. `scipy.ndimage.label` connected components in (ρ,θ), keep clusters ≥
   `min_cluster_size`, report per-cluster `median(θ)` and `median(ρ)·binning`.

This split — a generic SHT core plus a bespoke, statistically-framed astronomy
wrapper — is central to everything below.

---

## §1 — `pyhough` vs. common current Hough-based detectors

General architectural comparison against the mainstream family: the textbook SHT,
OpenCV `HoughLines`/`HoughLinesP`, scikit-image `hough_line`/
`probabilistic_hough_line`, and kernel/probabilistic variants. Not a
line-by-line algorithm comparison.

### 1.1 Architecture, aspect by aspect

| Aspect | `pyhough` (as used) | Typical current Hough detector |
|---|---|---|
| **Binary-image formation** | **Noise-based, mask-aware threshold** (`nsig·skyNoise`, bad planes removed), plus **template-ratio normalization** by valid path length | Usually an **edge map** (Canny/Sobel + thinning); no statistical noise model, no path-length normalization |
| **What votes** | Every above-threshold pixel of the **filled region** | Thinned **edge pixels** (1-px contours) |
| **Voting scheme** | Hard `+1`, nearest-`rho`, no smoothing | Same for SHT; kernel/probabilistic variants spread or subsample votes |
| **Accumulator** | Dense, `uint16`, `drho=1`, auto `ntheta` | Dense, typically `int32`; user-set `rho`/`theta` resolution |
| **Peak detection** | **None in library**; caller does threshold → connected-components → per-cluster medians | Usually **bundled**: threshold + non-max suppression, often ranked |
| **Output** | Infinite lines `(θ, ρ)` only; unordered; no width | SHT: infinite lines; probabilistic variants: **segments with endpoints**; KHT: ranked lines |
| **Preprocessing for faint/extended features** | Binning integrates low-surface-brightness flux before voting | Generally none; edge maps fragment on low-gradient features |
| **Complexity** | `O(N_set · N_theta)` brute force | Comparable for SHT; probabilistic/kernel variants cheaper |

### 1.2 Potential improvements (what `pyhough`'s approach does *better*)

1. **Statistically-motivated, mask-aware binary formation.** The "Good" of this
   design: the detection map is not a bare Canny output but a **noise-relative
   threshold** (`skyNoise` or MAD) with bad mask planes explicitly excluded.
   *Reasoning:* astronomical streaks are defined by significance over sky, not by
   local gradient; a σ-based threshold is directly interpretable and portable
   across exposures, whereas Canny hysteresis thresholds are scene-dependent and
   unphysical for this domain.

2. **Template-ratio (path-length) normalization.** Dividing votes by an all-ones
   transform corrects each (ρ,θ) cell for **how many valid pixels that line could
   traverse** — accounting for masked gaps and the image-edge foreshortening that
   makes short chords accumulate fewer votes.
   *Reasoning:* raw SHT peak height conflates "strong line" with "line that
   happens to cross more pixels." Normalization makes the peak statistic closer
   to a per-pixel occupancy fraction, which is fairer across orientations and
   robust to detector gaps — a correction absent from mainstream SHT/edge Hough.

3. **Region voting + pre-binning suits faint, extended streaks.** Voting with
   filled thresholded pixels (not thinned edges), after `16×` binning, integrates
   diffuse low-contrast flux.
   *Reasoning:* Canny edges on a faint satellite trail are fragmentary and noisy;
   region voting on a binned, SNR-boosted image captures the trail as a
   coherent set of voters. This is a legitimate advantage for low-surface-
   brightness linear features over edge-first pipelines.

4. **Minimal, transparent, dependency-light core.** A small C SHT plus a
   back-projection routine; deterministic and easy to reason about.
   *Reasoning:* fewer moving parts than kernel/probabilistic detectors means
   fewer parameters to mistune and simpler failure analysis — valuable for a
   detector meant to be one interchangeable core among several.

### 1.3 Potential regressions (what `pyhough`'s approach does *worse*)

1. **Filled-region voting admits compact-source contamination.** Because every
   above-threshold pixel votes — not just thinned edges — bright stars, galaxies,
   and saturated blobs dump many quasi-collinear votes across many (ρ,θ) cells.
   *Reasoning:* edge-thinning in mainstream pipelines suppresses compact,
   non-linear structure before voting; `pyhough` shifts that entire burden onto
   downstream global cuts (`min_streak_length`, ratio threshold), which cannot
   distinguish "long faint line" from "several bright sources that happen to
   align."

2. **`uint16` accumulator overflow risk.** Vote counts along a line are bounded
   by the number of set pixels on it; on large or heavily-lit *unbinned* inputs
   this can exceed 65535 and wrap silently.
   *Reasoning:* mainstream detectors use `int32` accumulators for exactly this
   reason. Binning (÷16) currently keeps counts small, so the risk is latent —
   but it is a correctness hazard if the core is used without the driver's
   binning.

3. **No built-in peak detection; correctness depends on ad-hoc post-processing.**
   The library returns only the vote map; the driver's peak step reports
   `median(θ[xs])` and `median(ρ[ys])` **independently** over each connected
   blob.
   *Reasoning:* independent medians of the two axes ignore the vote weights and
   the joint peak shape, so the recovered (ρ,θ) is a geometric blob-center, not a
   vote-weighted centroid — less precise than the NMS + smoothing that bundled
   detectors apply, and sensitive to blob morphology in parameter space.

4. **Hard nearest-`rho` binning, `drho=1`, no interpolation.** Localization is
   quantized at one accumulator cell.
   *Reasoning:* mainstream SHT has the same coarseness, but kernel/sub-pixel
   variants recover finer estimates; `pyhough` offers no path to sub-bin accuracy
   at the core, so precision is capped by `drho` and `ntheta`.

5. **Output is infinite lines with no extent, width, or score.** No endpoints,
   length, width, or per-line significance are returned.
   *Reasoning:* probabilistic Hough variants return segments directly; `pyhough`
   requires the caller to reconstruct extent (here, only indirectly via the
   template counts) and provides no ranking to prioritize detections.

6. **Brute-force `O(N_set · N_theta)` scaling.** Every set pixel is revisited for
   every angle.
   *Reasoning:* acceptable on binned images but poor on full-resolution large
   sensors; probabilistic and kernel detectors avoid scanning all pixels at all
   angles. The design leans on binning to stay tractable, trading resolution for
   speed.

7. **Maintenance fragility.** `pyhough_lib.py` uses `np.bool` (removed in
   NumPy ≥ 1.24; the current stack ships numpy 2.2.6, so this **will** raise
   `AttributeError`), and `simple_streak_finder.py`
   references an undefined `im` (should be `exposure`) at the `summaryStats`
   lookup.
   *Reasoning:* not architectural, but it signals the code is unmaintained and
   would not run as-is on a modern stack — relevant to any adoption decision.

---

## §2 — `pyhough` vs. KHT (direct comparison)

Now the specific case: `pyhough` against the current reference detector
[`lsst.kht`](../kht/) / [`KHTDetectTask`](../../../knowledge/kht-detect.md). KHT
is the *Kernel-Based* Hough transform (Fernandes & Oliveira 2008): edge linking
→ collinear-segment clustering → uncertainty-weighted Gaussian voting → smoothed,
non-max-suppressed peak detection.

### 2.1 Algorithmic differences

| Aspect | `pyhough` | KHT |
|---|---|---|
| **Primitive that votes** | Individual thresholded pixels | **Clusters of approximately-collinear edge pixels** (linking + Lowe subdivision) |
| **Vote shape** | Hard `+1`, nearest bin | **Elliptical-Gaussian kernel** whose covariance comes from per-cluster PCA + uncertainty propagation |
| **Preprocessing** | σ-threshold + mask removal + binning | Canny → 8-connected `find_chains` → recursive `find_clusters` (destroys input) |
| **Accumulator** | Dense `uint16`, `drho=1`, corner origin, θ∈[0,π) rad | Dense `int32`, `delta=0.5`, **image-center origin**, θ∈[0,180) deg |
| **Peak detection** | Caller: threshold → CC-label → medians | Built-in: **3×3 Gaussian convolution → descending sort → visited-neighbour NMS**, ranked by relevance |
| **Significance model** | **Noise-σ threshold on template-normalized ratio** | Relative `kernel_min_height` (+ LSST's `abs_kernel_min_height` bolt-on) |
| **Compact-source rejection** | Global cuts only (`min_streak_length`, ratio) | **Structural**: short/non-collinear chains never form clusters |
| **Complexity** | `O(N_set · N_theta)` over all set pixels | `O(edges)` + clustering; only surviving kernels vote |
| **Output** | Unordered `(θ, ρ)` infinite lines | `(ρ, θ)` **sorted by relevance** |

### 2.2 Potential improvements (`pyhough`'s decisions vs. KHT)

1. **A principled significance test that KHT lacks.** `pyhough`'s
   noise-σ-on-normalized-ratio directly answers "is there a real line?" via image
   statistics. KHT's culling is a *relative* kernel height — meaningless when no
   line exists, which is exactly why the LSST fork had to add
   `abs_kernel_min_height` (see [KHT README](../kht/) / fork notes).
   *Reasoning:* `pyhough`'s statistic is anchored to sky noise and valid path
   length, so its threshold has physical units and a null model; KHT's is
   scene-relative and required an empirical patch to behave when the scene is
   line-free. On the significance-modeling axis, `pyhough`'s decision is stronger.

2. **Path-length / mask normalization has no KHT analog.** KHT kernel height
   depends on the *cluster*, not on how many valid pixels the infinite line could
   cross; masked detector gaps and edge foreshortening are invisible to it.
   *Reasoning:* for LSST difference images riddled with masked regions,
   `pyhough`'s template ratio corrects the peak statistic for lost path, whereas
   KHT peaks are biased by unmodeled gaps.

3. **No edge detection required → robust to faint/low-gradient streaks.** KHT
   lives or dies by Canny: if edges fragment (faint trails), chains fall below
   `cluster_min_size` and no kernel votes. `pyhough` votes the region flux
   directly.
   *Reasoning:* the KHT front-end's collinearity structure is powerful on
   sharp-edged features but brittle on diffuse ones; `pyhough`'s region voting
   plus binning degrades more gracefully as SNR drops.

4. **Fewer core parameters.** No `cluster_min_deviation`, `delta` kernel
   modeling, or `n_sigmas` kernel spread to tune at the core.
   *Reasoning:* KHT's precision comes from a chain of coupled heuristics
   (link → subdivide → PCA → propagate → cull → smooth); each is a tuning
   surface. `pyhough`'s core has essentially one knob (`ntheta`), pushing tuning
   into a smaller, statistically-interpretable driver.

### 2.3 Potential regressions (`pyhough`'s decisions vs. KHT)

1. **No collinearity pre-filter → dirtier accumulator.** KHT's `find_chains` +
   `find_clusters` guarantee that only linked, approximately-straight pixel
   groups ever vote. `pyhough` votes all above-threshold pixels.
   *Reasoning:* KHT structurally rejects compact and curved structure *before*
   the accumulator; `pyhough` cannot, so its (ρ,θ) map carries a higher, more
   structured background that the global cuts must fight — a fundamentally weaker
   position on false-positive control.

2. **No uncertainty-aware voting → coarser, noisier peaks.** KHT spreads each
   cluster's vote as a Gaussian whose width encodes the fitted-line covariance,
   then convolves the accumulator with a 3×3 Gaussian before NMS. `pyhough` casts
   hard, quantized votes with no smoothing.
   *Reasoning:* KHT's kernel + smoothing denoise the parameter space and yield
   sub-`delta` peak localization; `pyhough`'s peaks are limited by `drho=1` and
   the caller's independent-median centroid, so both **precision and
   noise-robustness of the peak** are worse.

3. **No built-in, ranked peak detection.** KHT returns lines sorted by
   convolved-vote relevance with neighbour suppression; `pyhough` returns an
   unordered set from connected-component medians.
   *Reasoning:* ranking and NMS are exactly the parts that make a Hough output
   usable downstream (pick top-N, suppress duplicates). `pyhough` reimplements a
   weaker version of this in the driver and provides no relevance ordering.

4. **`uint16` + full-region voting vs. `int32` + culled kernels.** `pyhough`
   produces far more raw votes per cell than KHT (which only lets surviving
   kernels vote) while using a narrower integer type.
   *Reasoning:* the combination raises both the magnitude of counts and the
   overflow exposure relative to KHT's `int32` accumulator.

5. **Worse scaling on dense inputs.** `O(N_set·N_theta)` over all set pixels vs.
   KHT's edge-sparse, kernel-only voting (designed for real-time).
   *Reasoning:* on a full-resolution amp with many detected pixels `pyhough` must
   binning-downsample to stay tractable, sacrificing resolution that KHT retains
   by voting only sparse edge clusters.

6. **Frame-convention mismatch with the shared LFD template.** `pyhough` uses a
   **corner origin** and radians `[0,π)`; KHT (and therefore
   [`detector-task.md`](../../../knowledge/detector-task.md)'s shared convention)
   uses an **image-centered** frame with degrees.
   *Reasoning:* KHT's output already matches the image-centered `(ρ, θ[deg])`
   frame that `LineProfile`/`StreakAdapter` consume; `pyhough` would need an
   explicit frame shift, adding a well-known class of geometry bugs the project
   specifically warns about.

---

## §3 — Recommended changes to `pyhough` (for future testing; NOT implemented)

Derived from §1 and §2 — each item cites the reasoning there rather than
re-deriving it. These are directive enough to implement without re-reviewing the
codebase. **None are applied here.** Ordered roughly by priority.

### A. Correctness / "won't run on a modern stack" fixes (blocking)

1. **`pyhough/pyhough_lib.py:50` — replace `np.bool` with `bool`.** `np.bool` was
   removed in NumPy ≥ 1.24. Change `image.astype(np.bool)` → `image.astype(bool)`.
   (§1.3-7.)
2. **`simple_streak_finder.py` summaryStats block — fix the undefined `im`.**
   Lines ~54–57 call `im.getInfo()`/`im.getSummaryStats()`; `im` is never
   defined. Use `exposure.getInfo()` (the driver's actual argument). (§1.3-7.)
3. **Widen the accumulator to `uint32`.** In `pyhough/pyhough_pywrap.c`
   (`make_transform_image`) change `NPY_UINT16` → `NPY_UINT32`, and in
   `pyhough.h`/`pyhough.c` change the `unsigned short *data` vote buffer to
   `uint32_t`. Removes the silent-overflow hazard on unbinned/large inputs.
   (§1.3-2, §2.3-4.)

### B. Reduce false positives from compact sources (accumulator cleanliness)

4. **Add an optional collinearity/compactness pre-filter before voting.** In the
   driver, before building the numerator `det`, drop connected components that
   are compact (e.g. cut on area, or on elongation = major/minor axis of the
   component). This emulates KHT's structural rejection of non-linear structure
   without adopting full edge-linking. (§1.3-1, §2.3-1.)
5. **Optionally vote on a thinned/skeletonized mask.** Provide a switch to
   morphologically thin `det` (1-px medial axis) prior to `Hough`, so extended
   blobs contribute edge-like votes rather than filled-region votes. Keep the
   filled-region path available for the faint-streak regime (§1.2-3). (§1.3-1.)

### C. Peak localization, ranking, and output semantics

6. **Replace independent-axis medians with a vote-weighted centroid.** In the
   per-cluster loop, weight each (ρ,θ) cell in the connected component by its
   `ratio` (or raw vote) value and compute a joint weighted centroid, or fit a
   local parabola around the peak, instead of `median(θ[xs])`/`median(ρ[ys])`
   separately. (§1.3-3, §2.3-2.)
7. **Return a per-detection significance and sort by it.** Emit
   `(θ, ρ, score)` where `score = (ratio_peak − median)/MAD`, sorted descending,
   to match KHT's relevance ranking and enable top-N selection downstream.
   (§1.3-5, §2.3-3.)
8. **Emit finite segments, not just infinite lines.** Either use the existing
   `pyhoughback` back-projection or intersect each `(ρ,θ)` with the valid-pixel
   footprint to return endpoints + length. The per-line length is already
   implicitly available from `transform_template` at the detected cell. (§1.3-5.)

### D. Precision and frame alignment

9. **Allow sub-bin `rho` refinement.** Expose `drho < 1` (currently hardcoded to
   `1.0` in `Hough.__init__`) and/or add parabolic interpolation of the peak along
   the ρ axis to recover sub-pixel offset. (§1.3-4, §2.3-2.)
10. **Convert output to the shared image-centered `(ρ, θ[deg])` frame.** Add a
    conversion (or option) from `pyhough`'s corner-origin/radians frame to the
    image-centered degrees convention used by KHT and
    [`detector-task.md`](../../../knowledge/detector-task.md), so lines are
    directly comparable to KHT/`maskStreaks` and consumable by `LineProfile`.
    (§2.3-6.)

### E. Preserve the genuine strengths when integrating

11. **Keep the noise-σ threshold + template-ratio normalization as the
    significance model.** These are `pyhough`'s real advantages over both generic
    Hough (§1.2-1/2) and KHT (§2.2-1/2) — do not discard them when wrapping.
    Promote `nsig_det`, `nsig_streak`, `min_streak_length`, `binning`, and
    `min_cluster_size` to explicit config fields in any `astro_lfd` wrapper.
12. **Wrap as a shared-template detector task.** To evaluate `pyhough` on equal
    footing with KHT, implement `PyHoughDetectTask(pipeBase.Task)` with
    `run(table, exposure) -> Struct(streaks=SourceCatalog, ...)` reusing
    steps 1–2 and 4–5 of [`detector-task.md`](../../../knowledge/detector-task.md)
    and swapping only step 3 (the SHT + template-ratio core). This makes its
    output directly comparable via the existing
    `scripts/kht_maskstreaks_compare.py` harness. (Ties §3 back to the project;
    depends on items 8 and 10 for compatible output.)

---

**See also:** [KHT detector notes](../../../knowledge/kht-detect.md),
[shared detector-task template](../../../knowledge/detector-task.md),
[line geometry / ρ-θ convention](../../../knowledge/geom-line.md),
[ADRT design](../adrt/design.md) (a parallel non-Hough core evaluation).
