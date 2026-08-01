# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## Project Overview

**astro_lfd** is a Python library for **Linear Feature Detection (LFD)** in
astronomical images acquired by the **LSST Camera**. It is built as an
**extension to the LSST Science Pipelines stack**: the stack is a prerequisite
for running the package (see `CONTRIBUTING.md` for setup), and detectors are
formulated as standard LSST tasks operating on `lsst.afw` data products.

The goal is to detect linear features — e.g. satellite streaks, cosmic-ray
tracks, diffraction spikes, and other roughly-straight signals — by mapping the
image into a line parameter space and locating concentrations there.

`astro_lfd` provides a **suite of interchangeable detectors** sharing a common
LSST task template (see `knowledge/detector-task.md`), so a new method is added
by swapping only the line-finding core. Detectors implemented / planned:

- **Kernel Hough Transform (KHT)** — edge-based Hough voting; the current
  reference detector (`astro_lfd.meas.detectStreaks`).
- **Approximate Discrete Radon Transform (ADRT)** — line-integral transform via
  the [`adrt`](https://adrt.readthedocs.io) package; the first non-Hough
  detector (design in `docs/detectors/adrt/design.md`).
- Further methods (e.g. Line Segment Detector, Frangi vesselness, YOLO) fit the
  same interface.

- **Language:** Python (≥3.13), **src-layout** package (`src/astro_lfd/`).
- **Core dependencies:** `numpy`, `scipy`. Each detector backend is an
  **optional extra** (e.g. `adrt`, `lsst.kht`) so the core never hard-depends on
  one method; install with `pip install -e ".[adrt,dev]"`.
- **Domain:** astronomical image processing (LSST Science Pipelines context).
- **Stack coupling:** most subpackages depend on the LSST stack (`lsst.geom`,
  `lsst.afw.*`, `lsst.pipe.base`, `lsst.meas.algorithms`, …). Only `utils/` is
  kept stack-independent; where non-stack numpy/detector code must interoperate
  with the stack, the bridge is done with explicit to/from converter functions
  (as in `astro_lfd.utils.testdata`), not by taking a hard dependency in the core.

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
├── docs/                  # unified LFD_DESIGN.md + per-detector docs/detectors/<method>/
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
  `astro_lfd` on top with `pip install -e ".[dev]"` (add detector extras as
  needed, e.g. `".[kht,dev]"` or `".[adrt,dev]"`; `".[all]"` for everything).
  Full setup (including the Claude Code
  cache-loader and the conda `auto_activate_base` gotcha) is in
  **`CONTRIBUTING.md`**.
- Core runtime deps: `numpy`, `scipy`. Detector backends are **optional
  extras** (e.g. `adrt>=1.2.0`). Dev: `black`, `mypy`, `pytest`.
- Some experiments reference Rubin Observatory / LSST tooling
  (`lsst.afw.image`, `lsst.daf.butler`, `lsst.obs.lsst`, `mixcoatl`). These are
  **optional** and environment-specific; core LFD code should not hard-depend on
  them (see the lazy afw adapter in `astro_lfd.utils.testdata`).

Verify the environment before running code:

```bash
which python                                  # expect .../lsst-scipipe-*/bin/python
python -c "import adrt; print(adrt.__version__)"   # expect 1.2.0 (if adrt extra installed)
python -m pytest tests/ -q                    # expect 48 passed
```

### Shell startup is slow on the first call

The **first** Bash command in a session resolves `loadLSST.sh` +
`setup lsst_distrib` over NFS, which takes tens of seconds and can time out.
Don't interpret an early hang as a broken command — let it finish (or re-run)
once and it is cached. A cache-loader (`~/.claude/lsst-env-loader.sh`, wired via
`BASH_ENV` in `~/.claude/settings.json`) snapshots the resolved environment so
every subsequent call is fast, and self-heals when the `w_latest` target
changes. Force a rebuild with `refresh-lsst-env`. Full recipe and the conda
`auto_activate_base` gotcha are in **`CONTRIBUTING.md`** → "Speeding up the LSST
stack for Claude Code".

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

## Detectors

Every detector is a plain `lsst.pipe.base.Task` sharing one template — same
input contract, weights rule, output `SourceCatalog`, and coordinate handling —
differing only in the **line-finding core**. See `knowledge/detector-task.md`
for the shared template and `knowledge/INDEX.md` for the per-detector notes.

Current registry:

| Detector | Core | Backend dep | Notes / docs |
|---|---|---|---|
| **KHT** | Canny edges → `lsst.kht.find_lines` → cluster | `kht` extra (scikit-image, scikit-learn) + `lsst.kht` (stack) | reference detector; `knowledge/kht-detect.md`, `docs/detectors/kht/` |
| **ADRT** | line-integral transform → per-quadrant peak-finding | `adrt` extra (`adrt>=1.2.0`) | design in `docs/detectors/adrt/design.md`; API in `knowledge/adrt-api.md` |
| *future* | e.g. Line Segment Detector, Frangi vesselness, YOLO | its own extra | add a `knowledge/<method>-*.md` note + `docs/detectors/<method>/` |

Each backend is an **optional extra** so the core never hard-depends on one
method. Detector-specific grounding facts (exact signatures, array shapes,
gotchas) live in the knowledge base, **not** here — e.g. the ADRT input
contract and `(4, 2N-1, N)` output layout are in `knowledge/adrt-api.md`.

> When in doubt about a detector backend's API, inspect the installed package
> (`inspect.getdoc`, `inspect.signature`) rather than guessing — then record
> what you learn in the relevant knowledge-base note.

## Working Conventions

- Prefer small, testable functions over notebook cells for anything reusable.
- Keep the LFD core free of observatory-specific and detector-specific
  dependencies; isolate any Butler/LSST I/O behind a thin, optional adapter
  layer, and each detector backend behind its optional extra.
- Line-parameter geometry is easy to get subtly wrong (offset vs. angle axes,
  quadrant orientation, power-of-two padding, `rho`/`theta` conventions). Add
  assertions on array shapes and dtypes at boundaries, and document the
  coordinate convention used.
- When adding a nontrivial detector behavior, add a knowledge-base note (see
  below) so the finding is not re-derived later.

## Knowledge Base — Progressively Refined (`knowledge/`)

This section defines a **progressively refined knowledge base**: a curated,
append-and-refine store of hard-won facts about the LFD detectors, their backend
packages, and the astronomical domain, optimized for **quick, low-context
recall** during development. The goal is that a future session can load a
*small* amount of text and immediately be correct about a detector, instead of
re-reading source, re-running experiments, or re-deriving geometry.

### Why this exists

Detector backends have non-obvious contracts (e.g. the `adrt` package's
power-of-two square input and `(4, 2N-1, N)` quadrant layout; the KHT
image-centered `(rho, theta)` frame). Re-discovering these burns context and
invites subtle bugs. Capturing them once, tersely, pays off on every subsequent
turn.

### Structure

Notes fall into two groups — **shared** (apply across detectors) and
**per-detector** (one method's backend/design). Add per-detector notes as
`<method>-*.md` when a new detector lands.

```
knowledge/
├── INDEX.md          # one-line pointer per note; the only file always loaded
│   # shared
├── detector-task.md  # the shared LSST-task template every detector inherits
├── geom-line.md      # line geometry, rho/theta convention, StreakAdapter bridge
├── testdata.md       # synthetic LFD test images and how they feed a detector
│   # per-detector (ADRT)
├── adrt-api.md       # verified signatures, shapes, gotchas of the adrt package
│   # per-detector (KHT)
└── kht-detect.md     # the KHTDetectTask streak detector: config, I/O, convention
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

- Re-verify a detector's backend note (e.g. `adrt-api.md`) whenever that
  backend's installed version changes.
- Treat `INDEX.md` as the source of truth for what exists; keep it in sync when
  adding, splitting, or deleting notes.
