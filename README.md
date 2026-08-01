# astro_lfd

**Linear Feature Detection (LFD)** for astronomical images acquired by the LSST
Camera. `astro_lfd` detects roughly-straight signals — satellite streaks,
cosmic-ray tracks, diffraction spikes — by mapping the image into a line
parameter space and locating concentrations there.

It provides a **suite of interchangeable detectors** behind a shared LSST task
template, so a new method can be added by swapping only the line-finding core:

- **Kernel Hough Transform (KHT)** — edge-based Hough voting (the current
  reference detector).
- **Approximate Discrete Radon Transform (ADRT)** — line-integral transform via
  the [`adrt`](https://adrt.readthedocs.io) package (optional extra).
- Further methods (e.g. Line Segment Detector, Frangi vesselness, YOLO) can be
  added behind the same interface.

## Layout

```
src/astro_lfd/
├── geom/    # line geometry / coordinate conventions
├── meas/    # measurement of detected features
├── pipe/    # LSST Science Pipelines tasks / drivers
├── table/   # detection output tables
└── utils/   # helpers, incl. synthetic test-image generation (testdata.py)
docs/        # unified LFD_DESIGN.md + per-detector docs/detectors/<method>/
knowledge/   # progressively-refined knowledge base (read INDEX.md first)
notebooks/   # scratch notebooks for testing and development
tests/       # unit tests
```

## Install

```bash
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[kht,dev]"      # + KHT detector deps (scikit-image, scikit-learn)
pip install -e ".[adrt,dev]"     # + ADRT detector backend (adrt)
pip install -e ".[all]"          # every detector backend + dev tools
```

Detector cores that target the LSST Science Pipelines (KHT's task form) also
require the stack itself (`lsst.kht`, `lsst.afw`, …), which is not pip-installable
— see `CONTRIBUTING.md`. The extras above cover only the pip-installable parts, so
early numpy-level work can run without the stack.

## Test

```bash
python -m pytest tests/ -q
```

The optional `lsst.afw.image` FITS tests are skipped automatically when the LSST
stack is unavailable.
