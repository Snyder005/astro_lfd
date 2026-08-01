# Synthetic test images (`astro_lfd.utils.testdata`)

**When relevant:** generating LFD test images, or feeding their planes into a
detector — what arrays exist, their shapes/dtypes/units, and how they map
to a detector's expected input. (Detector-agnostic; ADRT's pow2-square
requirement is called out as a scoped aside.)

**Verified:** `pytest tests/test_testdata.py` (15 passed) against numpy 2.2.6,
scipy 1.16.3, astropy 7.2.0, `lsst.afw.image` present, on 2026-07-03.

## Data product: `TestImage`

Three co-registered planes plus a `meta` dict. Simulates one straight streak
(top-hat cross-section convolved with the PSF) on a background-subtracted
difference image.

| Plane | dtype | Units | Notes |
|---|---|---|---|
| `image` | `float32` | `meta['BUNIT']` (default `nJy`) | Background-subtracted; sky mean ≈ 0. |
| `variance` | `float32` | `BUNIT²` | Read noise² + sky + streak; **strictly > 0**. |
| `mask` | `int32` | bitmask | `0` = good. afw bit convention (`BAD`=bit0, …). Starts empty. |

Shape is `(ny, nx)`, default `(4004, 4096)` = LSST Camera science sensor
(rows × cols). These map to the shared LFD inputs `D` (image), `V` (variance),
`M` (mask) in `docs/LFD_DESIGN.md` §2.

## Array-input contract for a detector — READ THIS

Facts common to every detector:

- The detector operates on detected/significant pixels, but **`mask` starts
  empty** — the detector tasks need a `DETECTED` plane as the seed
  ([detector-task](detector-task.md)). Set it by thresholding the *noise-free*
  `streak_signal` at N·sigma, e.g. (as in the comparison harness
  `scripts/kht_maskstreaks_compare.py`):
  `detected = streak_signal(...) > 5*sqrt(read_noise**2 + sky + signal)`, then
  `exposure.mask.array[detected] |= mask.getPlaneBitMask("DETECTED")`.
- Detectors that transform the **significance image `S = D/√V`** (rather than the
  raw image) are insensitive to unit choice: `calib` is **invariant** under `S`
  (image → D/c, variance → V/c², so D/√V unchanged), so the output `unit`/`calib`
  does NOT affect detection, only output realism/compatibility. Don't "fix" units
  to chase detection changes; there are none.
- `float32` planes: cast to float as a numpy-level core expects; fine as-is.
  Poisson/normal draws are done in electrons then divided by `calib`.

**ADRT-specific aside — pow2-square padding.** The ADRT
(`knowledge/adrt-api.md`) additionally requires **square, power-of-two, float**
input. The test images are **neither square nor power-of-two** (4004×4096), so
the ADRT front-end MUST pad/tile before `adrt.adrt()`:

- Default `(4004, 4096)` → pad to `(8192, 8192)` (next pow2 square) with
  `numpy.pad(..., constant_values=0)`; record pad offsets to map coordinates
  back (`docs/detectors/adrt/design.md` §Step 0). 0-pad is safe: zeros add
  nothing to line sums.
- For fast unit tests / experiments, pass `shape=(256,256)` (or any pow2
  square) to `simulate_exposure` so output is ADRT-ready with no padding.
- The default shape is NOT ADRT-ready; KHT has no such constraint.

## Units (the notebook's open issue, resolved)

Radiometric sim is always in **electrons** (Poisson needs counts). Final planes
divided by scalar `calib` (e⁻ per output unit); variance by `calib²`.
- `unit='nJy'`, `calib`=photometric factor → matches calibrated LSST diff images (default).
- `unit='ADU'`, `calib`=amplifier gain.
- `unit='electron'`, `calib=1` → reproduces the original notebook exactly.
`calib` is a single frame-wide scalar — a documented simplification (a real CCD
has 16 per-amp gains).

## Minimal usage

```python
from astro_lfd.utils import testdata as td
ti = td.simulate_exposure(td.StreakConfig(theta=30.0, rho=2000.0, width=20.0,
                                          peak_signal=1000.0),
                          band="i", seed=42)          # default (4004,4096) nJy
td.save_npz(ti, "streak.npz")     # compact, no LSST dep
td.save_fits(ti, "streak.fits")   # lsst.afw.image.ExposureF, Pipelines-ready
# ADRT-ready small image (pow2 square, no padding needed):
small = td.simulate_exposure(td.StreakConfig(rho=128.0), shape=(256, 256), seed=0)
```

**Gotchas:**
- `StreakConfig.theta` is **degrees** (converted internally). Passing radians
  gives a wrong line — there is a regression test for this.
- `ExposureF(nx, ny)` takes **(width, height)**; arrays are `(ny, nx)`. Don't
  swap.
- Default shape is NOT ADRT-ready — pad or use a pow2-square `shape=`.
- `save_npz` uses `allow_pickle=False`; `meta` is stored as its `repr` and
  restored with `ast.literal_eval` (safe, no pickle).

**See also:** [detector-task](detector-task.md), [adrt-api](adrt-api.md),
[LFD design](../docs/LFD_DESIGN.md),
ADRT specifics `../docs/detectors/adrt/design.md`
