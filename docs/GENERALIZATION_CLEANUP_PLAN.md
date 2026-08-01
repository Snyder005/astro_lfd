# Generalization Cleanup Plan (ADRT-only → LFD framework)

**Status:** in progress. **Branch:** `feature/generalize-lfd-suite`.
**Scope:** documentation / knowledge / packaging cleanup **only — NO `src/*.py`
code changes.** Reframes the repo from "an ADRT project" to "an LFD framework"
where KHT (reference) and ADRT are peer detectors under a shared task template.

Save-point doc so a future session can resume without re-deriving state (created
because a prior session was interrupted by an API timeout).

## Directives locked with the user

1. **Docs tree:** keep `docs/LFD_DESIGN.md` as the **unified, detector-agnostic**
   design doc for the shared LFD framework. Create per-detector subtrees
   `docs/detectors/adrt/` and `docs/detectors/kht/`. Split detector docs into
   **relevant sections** (not one giant file per detector).
2. **Dependencies:** separate **per-detector** requirements (adrt deps vs kht
   deps) as optional extras; keep genuinely shared deps (numpy, scipy, …) in the
   core. The overall *environment* provides everything; the *requirements*
   separate by detector.
3. **Environment (tension resolved):** `pipe/` and `meas/` import
   `lsst.kht`/`lsst.afw`/`lsst.pipe.base`, which are **not pip-installable** —
   they come from the LSST stack. A standalone `conda create` cannot run the KHT
   detector or any task code. **Resolution:** the environment *is* the LSST stack
   (weekly tag, e.g. `w_2026_31`, cached via `~/.claude/lsst-env-loader.sh`), with
   detector deps layered on top via `pip install -e ".[adrt,kht,dev]"`. This is
   the single documented default runtime — robust for a forgetful future session.
   - "Run outside the stack" is still supported *for free at the packaging layer*:
     early numpy-level work (e.g. `utils/testdata.py` + `adrt`) can
     `pip install -e ".[adrt]"` in a plain venv. The stack is required only once a
     detector reaches the `pipe`/`meas` task form. Per-detector extras separate
     the requirements; the stack-based env is the default.

## Starting state (already done before this plan was written)

Working-tree edits already generalized (committed? NO — unstaged at plan time):
- `CLAUDE.md`, `README.md`, `pyproject.toml`, `src/astro_lfd/__init__.py` reframed
  ADRT-project → LFD-framework (KHT + ADRT peers). `pyproject.toml` already has an
  `adrt` extra but **no `kht` extra yet**.

## Steps

### 1. `pyproject.toml` — split requirements per detector
- Core stays `numpy`, `scipy` (shared).
- `adrt = ["adrt>=1.2.0"]` (already present).
- Add `kht = ["scikit-image", "scikit-learn"]` (the pip parts of the KHT core;
  `lsst.kht` + `lsst.afw` come from the stack — note this in a comment).
- Add convenience `all = ["astro_lfd[adrt,kht,dev]"]` or explicit union.
- Verified detector imports (from `grep` of src):
  - KHT (`meas/detectStreaks.py`): `skimage.feature`, `sklearn.cluster`,
    `lsst.kht`, `lsst.afw.*`, `lsst.pipe.base`, `lsst.meas.algorithms.maskStreaks`.
  - ADRT (design only, not yet coded): `adrt`.
  - shared/core: `numpy`, `scipy`; `utils/testdata.py` also uses `scipy.special`.

### 2. `docs/` restructure
- `docs/LFD_DESIGN.md`: strip "ADRT-based" title/framing → **unified** framework
  design (shared prepare→transform→detect→post-process→emit shape, inputs D/M/V,
  Hesse-form output, astronomy modifications common to all detectors). Move
  ADRT-specific method content OUT (see next).
- **New** `docs/detectors/adrt/design.md`: the ADRT specifics currently in
  `LFD_DESIGN.md` §2 (why ADRT vs KHT, quadrant `(4,2N−1,N)` peak-finding),
  §3 ADRT preconditions, §4 ADRT pipeline, §5 ADRT astro mods, §6 ADRT open
  questions, references. Split into sections if it grows.
- `git mv` KHT docs → `docs/detectors/kht/`:
  - `KHT_MASKSTREAKS_DISCREPANCY.md` → `maskstreaks-discrepancy.md`
  - `KHT_MASKSTREAKS_RESUME.md`      → `maskstreaks-resume.md`
  - `KHT_MASKSTREAKS_TESTPLAN.md`    → `maskstreaks-testplan.md`

### 3. `knowledge/` generalization
- `INDEX.md`: group **shared** vs **per-detector**; fix the two hooks that say
  "the ADRT detector" (testdata, detector-task) to be detector-agnostic.
- `detector-task.md`: retitle "KHT today, ADRT next" → detector-agnostic
  (KHT = validated reference; ADRT = one peer that swaps the core).
- `testdata.md`: reframe "feed the ADRT detector" → feed *a* detector; keep
  genuinely ADRT-specific facts (pow2 padding, D/√V) as clearly-scoped asides.
- `kht-detect.md`: minor — "template for future detectors (incl. the ADRT one)"
  → "…(e.g. ADRT)"; keep the ADRT-inherits-weight-rule note.
- Fix cross-refs to moved docs: `../docs/LFD_DESIGN.md` still valid; ADRT design
  refs now point to `../docs/detectors/adrt/design.md`.

### 4. `CLAUDE.md` / `README.md` / `CONTRIBUTING.md` sync
- `CLAUDE.md`: docs-layout line + registry already reference `docs/detectors/<method>.md`;
  make them match the actual tree (adrt/design.md, kht/…). Add `kht` extra to
  install examples.
- `README.md`: `docs/ (LFD_DESIGN.md)` layout line → new tree; add `kht` extra to
  install block.
- `CONTRIBUTING.md`: document the single stack-env + per-detector-extras recipe
  (`pip install -e ".[adrt,kht,dev]"`), the stack-free fallback for early numpy
  work, and expand the verify block (import adrt, skimage, sklearn, lsst.kht,
  lsst.afw, lsst.pipe.base; `pytest -q`).

### 5. Build + preserve the environment
- Source stack (cached loader), `pip install -e ".[adrt,kht,dev]"`.
- Verify: `python -c "import adrt, skimage, sklearn, lsst.kht, lsst.afw.image, lsst.pipe.base"`
  and `python -m pytest tests/ -q` (expect 15 passed).

## Cross-reference inventory (to keep in sync when moving files)

- `LFD_DESIGN.md` referenced by: `README.md:27`, `knowledge/detector-task.md:98`,
  `knowledge/testdata.md:24,34,83`.
- `KHT_MASKSTREAKS*` referenced by: `scripts/kht_maskstreaks_compare.py`
  (comment only — confirm before editing), `knowledge/kht-detect.md`, and the KHT
  docs cross-referencing each other.
- `docs/detectors/…` referenced by: `CLAUDE.md:25,52,164,165`.

## Guardrails
- **Do not modify `src/*.py`.** Only non-code touch allowed: a stale doc-path
  inside a *comment* in `scripts/kht_maskstreaks_compare.py` (verify it's a comment).
- No merge to `main`; feature branch + draft PR per project CLAUDE.md.

## Progress checklist
- [x] 1. pyproject per-detector extras (adrt, kht, dev, all)
- [x] 2. docs restructure (unified LFD_DESIGN + docs/detectors/adrt/design.md + docs/detectors/kht/*)
- [x] 3. knowledge generalization + cross-refs (INDEX grouped, detector-task/testdata/kht-detect reframed)
- [x] 4. CLAUDE/README/CONTRIBUTING sync (docs tree, kht extra, single-env recipe + fallback)
- [x] 5. env build + verify — `pip install -e ".[all]"` on lsst-scipipe-13.1.0;
       all imports OK; **pytest 48 passed** (docs updated 15→48; test_testdata.py alone is 15).

**COMPLETE (2026-07-31).** No `src/*.py` code changed. Only non-code touch:
doc-path strings in `scripts/kht_maskstreaks_compare.py` module docstring.
Remaining: commit on `feature/generalize-lfd-suite` + draft PR (not yet done).
