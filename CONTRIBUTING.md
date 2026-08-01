# Contributing to `astro_lfd`

## Development environment

`astro_lfd` is developed against the shared Rubin/LSST stack on SDF. The
optional `lsst.afw.image` code paths and the FITS I/O tests require it.

### Basic setup

```bash
source /sdf/group/rubin/sw/w_latest/loadLSST.sh
setup lsst_distrib
```

Then install `astro_lfd` on top of the stack in editable mode. Dependencies are
split per detector so you only pull in what a given detector needs (shared
`numpy`/`scipy` are always installed as core):

```bash
pip install -e ".[all]"          # every detector backend + dev tools (recommended)
# or pick per detector:
pip install -e ".[kht,dev]"      # KHT: scikit-image, scikit-learn (+ lsst.kht from the stack)
pip install -e ".[adrt,dev]"     # ADRT: adrt>=1.2.0
```

**Environment model.** The environment *is* the LSST stack (weekly tag), with
detector deps layered on top by the `pip install` above — this is the single,
reproducible default, because the task-form detectors (`pipe/`, `meas/`) import
stack-only modules (`lsst.kht`, `lsst.afw`, `lsst.pipe.base`,
`lsst.meas.algorithms`) that are **not** pip-installable. The per-detector extras
separate the *requirements*; the stack provides the shared *runtime*.

*Stack-free fallback:* the `astro_lfd` core is kept stack-independent, so early
numpy-level work (e.g. `utils.testdata` + a transform backend like `adrt`) can be
installed and run in a plain venv with just `pip install -e ".[adrt]"`. The stack
is required only once a detector reaches its LSST-task form.

### Verify

```bash
python -c "import numpy, scipy; print('core OK')"
python -c "import skimage, sklearn; print('kht deps OK')"    # if kht extra installed
python -c "import adrt; print('adrt', adrt.__version__)"     # if adrt extra installed; expect 1.2.0
python -c "import lsst.kht, lsst.afw.image, lsst.pipe.base; print('stack OK')"  # needs the stack
python -m pytest tests/ -q                                   # 48 passed
black --line-length 110 --check src/ tests/
mypy src/astro_lfd
```

The afw FITS tests are skipped automatically when `lsst.afw.image` is
unavailable (e.g. outside the stack), so the core suite still runs.

## Speeding up the LSST stack for Claude Code (optional)

Claude Code's Bash tool sources shell initialization on every command. Sourcing
`loadLSST.sh` + `setup lsst_distrib` over NFS on each call is slow and can time
out. The workaround is to resolve the environment **once**, cache it, and
re-source the snapshot on subsequent calls.

### 1. Cache-loader script

Create `~/.claude/lsst-env-loader.sh`:

```bash
# shellcheck shell=bash
# LSST env loader for Claude Code's Bash tool (wired via BASH_ENV in settings.json).
# Caches the resolved `setup lsst_distrib` env and re-sources it instead of re-resolving.
# Self-healing: rebuilds when the `w_latest` target changes; `refresh-lsst-env` forces it.

_LSST_CACHE="${HOME}/.cache/lsst-env.sh"
_LSST_LOAD_CMD='source /sdf/group/rubin/sw/w_latest/loadLSST.sh && setup lsst_distrib'

_lsst_build_cache() {
    local target
    target=$(readlink -f /sdf/group/rubin/sw/w_latest 2>/dev/null)
    mkdir -p "${HOME}/.cache"
    env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin bash --noprofile --norc -c '
        source /sdf/group/rubin/sw/w_latest/loadLSST.sh >/dev/null 2>&1
        setup lsst_distrib
        echo "# Auto-generated LSST env snapshot. Do not edit by hand."
        echo "# w_latest -> '"$target"'"
        export -p
        declare -f setup unsetup 2>/dev/null
    ' > "${_LSST_CACHE}.tmp" && mv -f "${_LSST_CACHE}.tmp" "$_LSST_CACHE"
}

refresh-lsst-env() {
    echo "Rebuilding LSST env cache (running real setup once)..." >&2
    _lsst_build_cache && { unset _LSST_ENV_LOADED; _lsst_load_env; echo "Done." >&2; }
}

_lsst_load_env() {
    local target
    target=$(readlink -f /sdf/group/rubin/sw/w_latest 2>/dev/null)
    [ "$_LSST_ENV_LOADED" = "$target" ] && return 0
    if [ ! -f "$_LSST_CACHE" ] || ! grep -q "w_latest -> ${target}$" "$_LSST_CACHE" 2>/dev/null; then
        _lsst_build_cache
    fi
    if [ -f "$_LSST_CACHE" ] && source "$_LSST_CACHE" 2>/dev/null; then
        export _LSST_ENV_LOADED="$target"
    else
        eval "$_LSST_LOAD_CMD"
        export _LSST_ENV_LOADED="$target"
    fi
}

_lsst_load_env
```

The loader sets up `lsst_distrib` generically; install `astro_lfd` once with
`pip install -e .` and it is importable in every cached shell thereafter.

### 2. Wire it into Claude Code

Add to `~/.claude/settings.json` (substitute your home path):

```json
"env": {
    "BASH_ENV": "/sdf/home/<u>/<user>/.claude/lsst-env-loader.sh"
}
```

### 3. Gotcha: conda base auto-activation shadows the stack

If you install miniforge/miniconda later, its `conda`/`mamba` init blocks in
`~/.bashrc` run **after** the loader and — with `auto_activate_base` on —
activate the conda `base` env on top of the stack. This drops
`lsst-scipipe/bin` from `PATH`, so `python` resolves to miniforge (no `afw`,
`pytest`, etc.) and tests silently skip.

Fix — give the stack precedence by disabling base auto-activation:

```bash
conda config --set auto_activate_base false
```

Verify a fresh shell keeps the stack: `which python` should point at
`.../lsst-scipipe-*/bin/python`. Use `conda activate base` when you deliberately
want miniforge.

## Git workflow

See `CLAUDE.md` → "Git Workflow": feature branches, draft PRs, atomic commits,
never commit directly to `main`.
