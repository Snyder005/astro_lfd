# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## Project Overview

**astro_lfd** is a Python library for **Linear Feature Detection (LFD)** in
astronomical images acquired by the **LSST Camera**. Its first detector is built
on the **Approximate Discrete Radon Transform (ADRT)**.

The goal is to detect linear features — e.g. satellite streaks, cosmic-ray
tracks, diffraction spikes, and other roughly-straight signals — by exploiting
the fact that a line in image space maps to a localized peak in Radon
(offset, angle) space.

- **Language:** Python (≥3.13), **src-layout** package (`src/astro_lfd/`).
- **Core dependency:** the [`adrt`](https://adrt.readthedocs.io) package
  (currently v1.2.0), which provides the ADRT and its inverse/adjoint.
- **Domain:** astronomical image processing (LSST Science Pipelines context).

## Repository Layout

```
astro_lfd/
├── src/
│   └── astro_lfd/
│       ├── geom/          # line geometry / coordinate conventions
│       ├── meas/          # measurement of detected features
│       ├── pipe/          # LSST Science Pipelines tasks / drivers
│       ├── table/         # detection output tables
│       └── utils/         # helpers, incl. testdata.py (synthetic images)
├── tests/                 # unit tests (pytest)
├── docs/                  # design docs (LFD_DESIGN.md)
├── knowledge/             # progressively refined knowledge base (see below)
├── notebooks/             # scratch notebooks for testing and development
├── pyproject.toml         # packaging + black/mypy/pytest config
├── CLAUDE.md              # this file
├── CONTRIBUTING.md        # LSST-stack environment setup
└── README.md
```

Most subpackages are currently scaffolding (empty `__init__.py`). Prefer
graduating stable code out of notebooks into these modules with tests.

## Environment & Dependencies

- Python ≥3.13. Developed against the shared Rubin/LSST stack on SDF; install
  `astro_lfd` on top with `pip install -e ".[dev]"`. Full setup (including the
  Claude Code cache-loader and the conda `auto_activate_base` gotcha) is in
  **`CONTRIBUTING.md`**.
- Runtime deps: `numpy`, `scipy`, `adrt`. Dev: `black`, `mypy`, `pytest`.
- Some experiments reference Rubin Observatory / LSST tooling
  (`lsst.afw.image`, `lsst.daf.butler`, `lsst.obs.lsst`, `mixcoatl`). These are
  **optional** and environment-specific; core LFD code should not hard-depend on
  them (see the lazy afw adapter in `astro_lfd.utils.testdata`).

Verify the environment before running code:

```bash
which python                                  # expect .../lsst-scipipe-*/bin/python
python -c "import adrt; print(adrt.__version__)"   # expect 1.2.0
python -m pytest tests/ -q                    # expect 15 passed
```

## Core Development Rules

1. Development Philosophy
  - **Simplicity**: Write simple, straightforward code
  - **Readability**: Make code easy to understand
  - **Performance**: Consider performance without sacrificing readability
  - **Maintainability**: Write code that's easy to update
  - **Testability**: Ensure code is testable
  - **Reusability**: Create reusable components and functions
  - **Less Code = Less Debt**: Minimize code footprint

2. Code Style
  - Modern type hints required for all code
  - PEP 8
  - Class names in PascalCase
  - Constants in UPPER_SNAKE_CASE
  - Document with docstrings
  - Line length: 110 chars maximum

3. Documenting Python APIs
  - PEP 257
  - Numpydoc Style 
  - Docstring line length: 79 chars maximum

## Python Tools

  - use `black` to fix PEP 8
  - use `mypy` for static type checking
  - use `pytest` for unit tests

## Git Workflow

- Always use feature branches; do not commit directly to `main`
  - Name branches descriptively: `fix/auth-timeout`, `documentation/code_examples`, `feature/api-pagination`
  - Keep one logical change per branch to simplify review and rollback
- Create pull requests for all changes
  - Open a draft PR early for visibility; convert to ready when complete
  - Ensure tests pass locally before marking ready for review
- Link issues
  - Before starting, reference an existing issue or create one
  - Use commit/PR messages like `Fixes #123` for auto-linking and closure
- Commit practices
  - Make atomic commits (one logical change per commit)
  - Prefer conventional commit style: `type(scope): short description`
    - Examples: `feature(eval): group OBS logs per test`, `fix(cli): handle missing API key`
  - Squash only when merging to `main`; keep granular history on the feature branch
- Practical workflow
  1. Create or reference an issue
  2. `git checkout -b feature/issue-123-description`
  3. Commit in small, logical increments
  4. Open a draft PR early
  5. Convert to ready PR when functionally complete and tests pass
  6. Never merge automatically, always prompt first

## ADRT Package — Essential Facts

Grounding facts (verified against installed v1.2.0). Keep these accurate; if the
installed version changes, re-verify.

- **Input contract:** `adrt.adrt(a)` requires a **square** image whose side
  length `N` is a **power of two**, dtype float. An optional leading **batch**
  dimension is allowed (2-D or 3-D input). Pad with `numpy.pad` if needed.
- **Output shape:** for input size `N`, each batch element yields shape
  `(4, 2*N-1, N)` — four **quadrants**, each spanning π/4 of angle. The
  `2*N-1` axis is **offset**; the `N` axis is **angle**.
- **Key functions:**
  - `adrt.adrt(a)` — forward ADRT.
  - `adrt.iadrt(a)` — exact inverse.
  - `adrt.iadrt_fmg(a, *, max_iters=None)` — iterative (full multigrid) inverse.
  - `adrt.bdrt(a)` — back-projection / adjoint.
  - `adrt.utils.stitch_adrt(a, remove_repeated=False)` — assemble the four
    quadrants into one contiguous image (useful for visualization).
  - `adrt.utils.unstitch_adrt(a)` — inverse of stitch.
  - `adrt.utils.coord_adrt(N)` — map ADRT indices to `(offset, angle)` physical
    coordinates. Central to interpreting detected peaks as image-space lines.
  - `adrt.utils.interp_to_cart(...)` — interpolate ADRT output to Cartesian
    (θ, s) sinogram coordinates.
  - `adrt.utils.truncate(...)`, `coord_cart_to_adrt(...)` — coordinate/shape helpers.
  - `adrt.core.*` — lower-level stepwise primitives (`adrt_step`, `num_iters`,
    `threading_enabled`, …); usually not needed directly.

> When in doubt about the ADRT API, inspect the installed package
> (`inspect.getdoc`, `inspect.signature`) rather than guessing — then record
> what you learn in the knowledge base below.

## Working Conventions

- Prefer small, testable functions over notebook cells for anything reusable.
- Keep the LFD core free of observatory-specific dependencies; isolate any
  Butler/LSST I/O behind a thin, optional adapter layer.
- Radon/ADRT geometry is easy to get subtly wrong (offset vs. angle axes,
  quadrant orientation, power-of-two padding). Add assertions on array shapes
  and dtypes at boundaries, and document the coordinate convention used.
- When adding a nontrivial ADRT-related behavior, add a knowledge-base note
  (see below) so the finding is not re-derived later.

## Knowledge Base — Progressively Refined (`knowledge/`)

This section defines a **progressively refined knowledge base**: a curated,
append-and-refine store of hard-won facts about the `adrt` package and the LFD
domain, optimized for **quick, low-context recall** during development. The
goal is that a future session can load a *small* amount of text and immediately
be correct about the ADRT, instead of re-reading source, re-running
experiments, or re-deriving geometry.

### Why this exists

The `adrt` package has non-obvious contracts (power-of-two square input, the
`(4, 2N-1, N)` quadrant layout, offset/angle axis conventions). Re-discovering
these burns context and invites subtle bugs. Capturing them once, tersely, pays
off on every subsequent turn.

### Structure

```
knowledge/
├── INDEX.md          # one-line pointer per note; the only file always loaded
├── adrt-api.md       # verified signatures, shapes, gotchas of the adrt package
├── adrt-geometry.md  # offset/angle conventions, quadrants, coord mappings
├── lfd-design.md     # detector design decisions, thresholds, peak-finding
└── astro-domain.md   # astronomical specifics (streaks, PSF, masking, units)
```

- **`INDEX.md` is the entry point.** It holds one line per note:
  `- [topic](file.md) — <hook describing when this is relevant>`.
  Read `INDEX.md` first; open a detailed note only when its hook matches the
  task. This keeps baseline context small.
- Each note is **one focused topic**, short enough to load cheaply.

### How to populate it (recommendations)

Populate the knowledge base to maximize **signal per token** and **recall
precision**. Follow these guidelines:

1. **One fact/topic per note; keep notes short.** A note should be scannable in
   seconds. If a note grows past ~1 screen, split it. Small notes mean an agent
   loads only what's relevant.
2. **Lead with a one-line summary / hook.** The first line should state when the
   note matters (this line is what goes in `INDEX.md`). Recall is driven by
   matching the task to hooks, so make hooks specific and searchable.
3. **Prefer verified, concrete facts over prose.** Record exact signatures,
   array shapes, dtype/size constraints, and error messages — things that are
   costly to re-derive. Mark each with *how it was verified* and *against which
   `adrt` version*, so staleness is detectable.
4. **Capture gotchas and "why", not just "what".** Note the mistake that was
   made and the reason for the resolution (e.g. "offset axis is `2N-1`, not
   `N` — do not confuse with angle"). The rationale prevents regressions.
5. **Write it the moment it's learned.** When you inspect the package, debug a
   shape mismatch, or settle a design question, immediately add/refine a note.
   Don't defer — the context is cheapest to capture right after discovery.
6. **Refine and dedupe, don't just append.** Before adding, check for an
   existing note on the topic and update it in place. Delete facts that are
   proven wrong. The base should get *sharper* over time, not merely longer.
7. **Cross-link with `[topic](file.md)`.** Link related notes so following one
   thread surfaces adjacent context.
8. **Keep it self-contained and copy-pasteable.** Include minimal runnable
   snippets (input shape → call → output shape) that an agent can reuse
   verbatim without opening the package source.
9. **Separate durable from speculative.** Verified API facts (`adrt-api.md`)
   should be stable; evolving design opinions (`lfd-design.md`) can change —
   date design decisions so their currency is clear.
10. **Record what is NOT true / dead ends.** Note approaches that were tried and
    abandoned (and why) to stop future sessions from repeating them.

### Note template

```markdown
# <topic>

**When relevant:** <one-line hook — copy this into INDEX.md>

**Verified:** <how / adrt version / date>

<terse facts, exact shapes/signatures, minimal snippet>

**Gotchas:** <mistakes to avoid and why>

**See also:** [related](other-note.md)
```

### Maintenance

- Re-verify `adrt-api.md` / `adrt-geometry.md` whenever the installed `adrt`
  version changes.
- Treat `INDEX.md` as the source of truth for what exists; keep it in sync when
  adding, splitting, or deleting notes.
