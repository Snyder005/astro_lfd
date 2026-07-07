# adrt geometry — quadrants, offset/angle axes, coord_adrt

**When relevant:** interpreting `adrt.adrt` output cells as physical lines —
which quadrant/axis is angle vs offset, angle ranges per quadrant, offset units,
and mapping peaks to `(offset, angle)`. Read before writing ADRT peak detection.

**Verified:** direct experiment against installed `adrt` **v1.2.0** (Python 3.13)
on 2026-07-07 — `coord_adrt`, plus injected single-line images to confirm which
quadrant/angle a known line lands in. Re-verify if `adrt` version changes.

## Accumulator layout (unstitched, from `adrt.adrt`)

```
A = adrt.adrt(a)          # shape (4, 2N-1, N), float
#   axis 0 : quadrant  (4 angle bands, each 45°)
#   axis 1 : offset    (2N-1)  — "height"/intercept
#   axis 2 : angle     (N)     — "slope"
```

Each cell = sum of `a` along one digital line. A line in image space → **one**
localized peak (ridge) here. **Detect peaks per quadrant on this unstitched
array** — never on the `stitch_adrt` mosaic (seams create false maxima and the
offset scaling differs across the band).

## coord_adrt(N) — cell → physical (offset, angle)

```python
c = adrt.utils.coord_adrt(N)     # ADRTCoord(offset, angle)
c.offset   # (4, 2N-1, N)  float64 — normalized offset, broadcastable to A
c.angle    # (4, 1,   N)  float64 — angle [radians], broadcastable to A
```

- `angle` is **squeezed** on the offset axis (shape `(4,1,N)`): angle depends only
  on quadrant + angle-index, not on offset. Use `np.broadcast_to` if a full array
  is needed; otherwise slice — cheaper.
- Cache `coord_adrt(N)` per `N` (it is pure); slice it in both peak detection and
  window extraction so geometry is derived once.

## Angle ranges per quadrant (VERIFIED, degrees)

Angles are in radians in `[-π/2, +π/2]`. Per quadrant, along the angle axis
(`angle_idx = 0 … N-1`):

| Quadrant | angle_idx=0 | angle_idx=N-1 | band |
|---|---|---|---|
| **q0** | −90° | −45° | [−90, −45] |
| **q1** |   0° | −45° | [−45,   0] |
| **q2** |   0° | +45° | [  0, +45] |
| **q3** | +90° | +45° | [+45, +90] |

So the four quadrants tile `[-90°, +90°]`, but **not** in monotonic index order:
q1 and q3 run "inward" toward 45° (q0/q1 meet at −45°, q2/q3 meet at +45°, q1/q2
meet at 0°). Angle sampling is **non-uniform** (denser near ±90°): e.g. N=8 q2
gives `0, 8.13, 15.95, 23.20, 29.74, 35.54, 40.60, 45°`.

Confirmed by injection (N=8): a line with image-`theta` (Hesse normal angle)
lands at:
- `theta=0°`   (vertical line)  → q0, angle_idx 0  (−90°)
- `theta=90°`  (horizontal)     → q1, angle_idx 0  (  0°)
- `theta=45°`                    → q2, angle_idx N-1 (+45°)
- `theta=-45°`                   → q0, angle_idx N-1 (−45°)

⇒ the ADRT `angle` is the digital-line direction; it differs from the image
Hesse `theta` by the usual 90° normal-vs-direction relation. **Establish the exact
image-`(rho,theta)` ↔ ADRT-`(offset,angle)` map with a ground-truth injection test
before trusting conversions** (planned test 6a.1 in
[peak-detection plan](../docs/ADRT_PEAK_DETECTION_PLAN.md)).

## Offset is NORMALIZED, not pixels (gotcha)

`c.offset` is dimensionless, range ≈ **[−1.5, 1.5]** (grows slowly toward ±1.5 as
N increases: max ≈ 1.312 at N=8, 1.406 at N=16, 1.453 at N=32). It is **not** a raw
pixel intercept. Do not treat an offset value as pixels — convert to the image
Hesse `rho` (in the padded/original frame) as an explicit, separate step, undoing
the Step-0 pad/shift (`docs/LFD_DESIGN.md`). Along axis 1 the offset varies
monotonically for fixed angle (see the per-column dumps in the verification).

## Nominal line length per cell

For per-line normalization / occupancy (flagging fragmented ridges), get the
nominal digital-line length of every cell in one shot:
`L_full = adrt.adrt(np.ones((N, N)))` — robust to geometry subtleties, no closed
form needed. Normalize peak values by `A_count = adrt.adrt((~M).astype(float))`
(valid-pixel count) and/or compare to `L_full`.

**Gotchas:**
- Offset axis is `2N-1`, angle axis is `N` — don't transpose.
- Quadrants are separate 45° bands with independent geometry — detect per quadrant.
- `angle` array is `(4,1,N)` (squeezed) — broadcast, don't assume `(4,2N-1,N)`.
- Offset is normalized (~[−1.5,1.5]), NOT pixels.
- ADRT `angle` ≠ image Hesse `theta`; verify the map with an injection test.

**See also:** [adrt-api](adrt-api.md),
[peak-detection plan](../docs/ADRT_PEAK_DETECTION_PLAN.md),
[detector-task](detector-task.md), `../docs/LFD_DESIGN.md`
</content>
