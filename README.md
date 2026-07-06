# astro_lfd

**Linear Feature Detection (LFD)** for astronomical images acquired by the LSST
Camera. `astro_lfd` detects roughly-straight signals — satellite streaks,
cosmic-ray tracks, diffraction spikes — by exploiting the fact that a line in
image space maps to a localized peak in Radon `(offset, angle)` space, computed
via the [Approximate Discrete Radon Transform (ADRT)](https://adrt.readthedocs.io).

## Layout

```
src/astro_lfd/
├── geom/    # line geometry / coordinate conventions
├── meas/    # measurement of detected features
├── pipe/    # LSST Science Pipelines tasks / drivers
├── table/   # detection output tables
└── utils/   # helpers, incl. synthetic test-image generation (testdata.py)
docs/        # design docs (LFD_DESIGN.md)
knowledge/   # progressively-refined knowledge base (read INDEX.md first)
notebooks/   # scratch notebooks for testing and development
tests/       # unit tests
```

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
python -m pytest tests/ -q
```

The optional `lsst.afw.image` FITS tests are skipped automatically when the LSST
stack is unavailable.
