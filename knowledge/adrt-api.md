# adrt package API

**When relevant:** calling the `adrt` package — input requirements, function signatures, output shapes.

**Verified:** `inspect.signature` / `inspect.getdoc` against installed `adrt` **v1.2.0**, Python 3.13, on 2026-07-02.

## Input contract (forward ADRT)

- `a` must be a **square** image, side length `N` a **power of two**, float dtype.
- Optional leading **batch** dimension → input is 2-D `(N, N)` or 3-D `(B, N, N)`.
- Pad with `numpy.pad` if the image is not a power-of-two square.

## Output shape

- For input size `N`, each batch element → shape **`(4, 2*N-1, N)`**.
- Axis 0: the **4 quadrants**, each spanning π/4 of angle.
- Axis 1 (`2N-1`): **offset**.
- Axis 2 (`N`): **angle**.

## Core functions

```python
adrt.adrt(a)                       # forward ADRT  -> (…, 4, 2N-1, N)
adrt.iadrt(a)                      # exact inverse
adrt.iadrt_fmg(a, *, max_iters=None)  # iterative full-multigrid inverse
adrt.bdrt(a)                       # back-projection / adjoint
```

## utils

```python
adrt.utils.stitch_adrt(a, remove_repeated=False)  # 4 quadrants -> contiguous image (viz)
adrt.utils.unstitch_adrt(a)                        # inverse of stitch
adrt.utils.coord_adrt(N) -> (offset, angle)        # ADRT index -> physical (offset, angle)
adrt.utils.interp_to_cart(...)                     # ADRT output -> Cartesian (theta, s) sinogram
adrt.utils.coord_cart_to_adrt(...)                 # Cartesian -> ADRT coords
adrt.utils.truncate(...)                           # shape helper
# dataclasses: ADRTCoord, ADRTIndex
```

`adrt.core.*` exposes stepwise primitives (`adrt_step`, `adrt_iter`, `num_iters`,
`threading_enabled`, …) — rarely needed directly.

**Gotchas:**
- Non-power-of-two or non-square input raises — always pad first.
- Don't confuse the offset axis (`2N-1`) with the angle axis (`N`).
- Re-verify this note if the installed `adrt` version changes.
