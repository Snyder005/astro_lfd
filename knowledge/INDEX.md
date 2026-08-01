# Knowledge Base Index

Read this file first. Open a detailed note only when its hook matches the task.
See `../CLAUDE.md` → "Knowledge Base" for how to populate and maintain these.

Notes are grouped **shared** (apply across all detectors) and **per-detector**
(one method's backend/design). Add per-detector notes as `<method>-*.md` when a
new detector lands.

## Shared

- [detector-task](detector-task.md) — reusable LFD detector-task template (input contract, weights rule, output format, coord handling) that every detector inherits by swapping only the line-finding core.
- [geom-line](geom-line.md) — line geometry primitives: `rho`/`theta` (Hesse) convention, `Line2D`/`LineSegment2D` API, weighted fit, and the afw `StreakAdapter` bridge.
- [testdata](testdata.md) — synthetic LFD test images: plane shapes/dtypes/units and how they feed a detector (`DETECTED` seed, `D/√V`; ADRT padding as a scoped aside).

## Per-detector — KHT

- [kht-detect](kht-detect.md) — the `KHTDetectTask` streak detector: config, `run` I/O, output fields, coordinate convention, and its validated relationship to `maskStreaks`.

## Per-detector — ADRT

- [adrt-api](adrt-api.md) — signatures, input contract, and output shapes of the `adrt` package.
