# ADRT multi-peak detection

**When relevant:** calling or tuning `ADRTDetectTask._find_peaks`, interpreting
its `AdrtPeak` descriptors, or converting ADRT indices to lines.

**Verified:** simulations in `tests/test_adrt_peaks.py` (N=1024, unit noise,
11 scenes x 2 seeds; N=2048 timing) on 2026-09-12 against `adrt` v1.2.0.

## Call contract

```python
task = ADRTDetectTask()
lines = task.detect(image)                       # square, power-of-two, float
lines = task._find_peaks(adrt.adrt(image))       # list[Line2D], ranked by S
lines, peaks = task._find_peaks(acc, return_peaks=True)
```

- Input: the raw `(4, 2N-1, N)` accumulator. The line-length accumulator
  `adrt(ones)` is computed and cached per `N` unless passed as `line_length`.
- Output: `Line2D` objects in the **PIXEL frame of the ADRT grid**
  (`arr[j, i] <-> (x=i, y=j)`), theta canonicalised to `[0, pi)`. Undoing
  binning/padding is the caller's job.
- `AdrtPeak(q, h, s, h_refined, s_refined, value, line)`: integer indices,
  refined indices, significance `S = A / sqrt(L)` in noise-sigma units.
- All tunables are keyword arguments (not Config). Defaults recover streaks
  down to ~0.5 sigma/px over 900 px and reject pure noise; see
  `src/astro_lfd/algorithms/README.md` for the argument table and rationale.
- `refine="butterfly"` is experimental and wrong at unit per-pixel noise;
  leave the default `"parabolic"`.

## Coordinate maps (`adrtButterfly.py`)

- `_adrt_to_hesse(q, h, s, N) -> (rho, theta)` closed form, accepts fractional
  `h`, `s`; matches `adrt.utils.coord_adrt` to 1e-9.
- `_hesse_to_adrt(rho, theta, N, quadrant=None) -> (q, h, s)`; with
  `quadrant=q` it extrapolates the line into that quadrant's frame (indices may
  fall outside `[0, 2N-1) x [0, N)`), which is how wing suppression follows a
  streak's butterfly across a seam.
- Quadrant theta bands (line-normal angle): `[0, pi/4) -> 3`, `[pi/4, pi/2) -> 2`,
  `[pi/2, 3pi/4) -> 1`, `[3pi/4, pi) -> 0`. Columns `s = 0` and `s = N-1` are
  seams shared with the neighbouring quadrant, so a seam line appears twice
  and is de-duplicated in Hesse space.

## Gotchas

- `S` is only unit-variance if the input image has unit white noise and zero
  background; scale the image by `1/sqrt(V)` and subtract sky first.
- A faint streak crossing a much brighter one is suppressed when its `S` is
  below the wing envelope (about `sin(dW)/sin(d)` of the bright peak plus 4
  sigma). At N=1024: 1 sigma/px next to 10 sigma/px is lost, 1.5 sigma/px is kept.
- `ADRTDetectTask` needs `self.timings = {}` (set in `__init__`) for the
  `@timed` decorators; `preprocess`/`postprocess` are still unfinished.
