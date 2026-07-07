# Knowledge Base Index

Read this file first. Open a detailed note only when its hook matches the task.
See `../CLAUDE.md` → "Knowledge Base" for how to populate and maintain these.

- [adrt-api](adrt-api.md) — signatures, input contract, and output shapes of the `adrt` package.
- [testdata](testdata.md) — synthetic LFD test images: plane shapes/dtypes/units and how they feed the ADRT detector (padding, `D/√V`).
- [geom-line](geom-line.md) — line geometry primitives: `rho`/`theta` (Hesse) convention, `Line2D`/`LineSegment2D` API, weighted fit, and the afw `StreakAdapter` bridge.
- [detector-task](detector-task.md) — reusable LFD detector-task template (input contract, weights rule, output format, coord handling) that the ADRT detector inherits by swapping only the line-finding core.
- [kht-detect](kht-detect.md) — the `KHTDetectTask` streak detector: config, `run` I/O, output fields, coordinate convention, and its validated relationship to `maskStreaks`.
