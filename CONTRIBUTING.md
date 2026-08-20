# Contributing to `astro_lfd`

## Development environment

`astro_lfd` is developed against the shared Rubin/LSST stack on SDF. The
optional `lsst.afw.image` code paths and the FITS I/O tests require it.

### Environment model (read this first)

The shared LSST stack under `/sdf/group/rubin/sw/` is **read-only** and updates
weekly — you are a stack *user* and cannot add packages to it. So the working
environment is built in two orthogonal layers on top of a **clean login
baseline** (`.bashrc` sets your profile, **no conda env activated, no stack
sourced**):

1. **The stack** — sourced per shell (`loadLSST.sh` + `setup lsst_distrib`). This
   *is* the runtime: it provides Python 3.13, numpy/scipy, and the stack-only
   modules the task-form detectors import (`lsst.kht`, `lsst.afw`,
   `lsst.pipe.base`, `lsst.meas.algorithms`) — none of which are pip-installable.
2. **`astro_lfd` + detector extras** — installed **once** into your
   **user** site (`pip install --user -e ...`), which the stack Python picks up
   automatically. These are *additions* you own; they never touch the stack.

A correctly set-up shell looks like this (versions track the current weekly):

```text
which python  -> /sdf/group/rubin/sw/conda/envs/lsst-scipipe-<ver>/bin/python
python -V     -> Python 3.13.x
numpy         -> 2.2.x
python -c "import lsst.afw.image"   -> imports cleanly
```

If instead `which python` is `/usr/bin/python` (or numpy is ~1.23), the stack is
**not** actually active — see "Ordering gotcha" below. This is the #1 setup
failure mode here.

### Setup — the user (manual, per new checkout)

From a clean login shell:

```bash
# 1. Source the stack FIRST, before touching conda (ordering matters — see below).
source /sdf/group/rubin/sw/w_latest/loadLSST.sh
setup lsst_distrib

# 2. Confirm the stack actually won (guard against a fragmented env).
which python                         # must be .../lsst-scipipe-*/bin/python
python -c "import lsst.afw.image; print('stack OK')"

# 3. Install astro_lfd + all detector extras into your USER site, editable.
cd /sdf/data/rubin/user/<u>/<...>/astro_lfd
pip install --user -e ".[all]"
```

`--user` is required: the stack site is read-only, so installs go to
`~/.local/lib/python3.13/site-packages`. Because that path is **pinned to the
stack's Python minor version (3.13)**, a future weekly that bumps Python will
need a one-time reinstall (step 3).

**Why `[all]`, and why extras exist.** Install `".[all]"` as the default — it
pulls every detector backend + dev tools. The per-detector extras (`[kht]`,
`[adrt]`, …, defined in `pyproject.toml`) exist **only to label which
requirement is unique to which detector**, so an obsolete detector's deps can be
culled later without disturbing the others. They are *bookkeeping*, not an
install-minimization strategy for day-to-day work.

```bash
pip install --user -e ".[all]"       # recommended default (every backend + dev)
# per-detector, only if deliberately isolating one method's requirements:
pip install --user -e ".[kht,dev]"   # KHT: scikit-image, scikit-learn (+ lsst.kht from the stack)
pip install --user -e ".[adrt,dev]"  # ADRT: adrt>=1.2.0
```

*Stack-free fallback:* the `astro_lfd` core is kept stack-independent, so early
numpy-level work (e.g. `utils.testdata` + a transform backend like `adrt`) can be
installed and run in a plain venv with just `pip install -e ".[adrt]"`. The stack
is required only once a detector reaches its LSST-task form.

### Setup — a Claude Code session (via the cache loader)

A Claude session starts from the same clean baseline but **must not** re-source
`loadLSST.sh` on every Bash call (slow over NFS, times out). Instead it relies on
the cache loader wired via `BASH_ENV` (see "Speeding up the LSST stack" below).
The per-session procedure is:

```bash
# The loader auto-sources the cached stack env. Verify it took:
which python                         # expect .../lsst-scipipe-*/bin/python
python -c "import numpy, lsst.afw.image; print(numpy.__version__, 'stack OK')"
```

If that shows `/usr/bin/python` / numpy ~1.23 / `No module named 'lsst'`, the
**cache is poisoned** (fragmented env) — rebuild it once and re-verify:

```bash
refresh-lsst-env
```

The `astro_lfd` editable install (user-site, step 3 above) persists across cached
shells, so a session does **not** reinstall — it only needs the stack to be
active. If `import astro_lfd` fails while the stack *is* active, it simply hasn't
been installed yet; ask the user before running a `--user` install on their
behalf.

### Verify

There is no full pytest suite yet. Primary verification is that the Python
version and all dependency imports resolve at their stack-linked versions:

```bash
python -V                                                    # 3.13.x
python -c "import numpy, scipy; print(numpy.__version__)"    # numpy 2.2.x
python -c "import skimage, sklearn; print('kht deps OK')"    # if kht extra installed
python -c "import adrt; print('adrt', adrt.__version__)"     # if adrt extra installed; expect 1.2.0
python -c "import lsst.kht, lsst.afw.image, lsst.pipe.base; print('stack OK')"  # needs the stack
python -c "import astro_lfd; print('astro_lfd OK')"          # user-site editable install
black --line-length 110 --check src/                         # lint
mypy src/astro_lfd                                           # type-check
```

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

### 3. Ordering gotcha: source the stack FIRST, and keep miniforge out of the way

The single most common failure here is a **fragmented environment**: `python`
resolves to `/usr/bin/python` (or a miniforge python) with old numpy (~1.23) and
**no `lsst` module**, even though you "sourced the stack". A correctly active
stack is Python 3.13 / numpy 2.2.x / `import lsst.afw.image` clean, at
`.../lsst-scipipe-*/bin/python`.

Two interacting causes:

- **conda base auto-activation.** If miniforge/miniconda `conda`/`mamba` init
  blocks in `~/.bashrc` run and `auto_activate_base` is on, they activate the
  conda `base` env **on top of** the stack, dropping `lsst-scipipe/bin` from
  `PATH`. Fix — give the stack precedence:

  ```bash
  conda config --set auto_activate_base false
  ```

- **Ordering + the moved miniforge3.** The user's miniforge3 was relocated and is
  now reached via a **symlink**. Order matters: once `setup lsst_distrib` has run,
  `conda` points at the **stack's own conda**, not miniforge3 — so running conda
  commands *after* sourcing the stack, or sourcing in the wrong order, can leave a
  half-built `PATH`. **Always source the stack first**, and don't `conda
  activate` on top of it unless you deliberately want miniforge (then use
  `conda activate base` explicitly).

Verify a fresh shell keeps the stack:

```bash
which python                                   # -> .../lsst-scipipe-*/bin/python
python -c "import numpy, lsst.afw.image; print(numpy.__version__)"   # -> 2.2.x
```

### 4. Gotcha: a stale cache only self-heals on a weekly-tag change

The loader rebuilds its cache when `readlink -f /sdf/group/rubin/sw/w_latest`
changes (i.e. a new weekly). It does **not** notice other environment shifts —
e.g. the miniforge3 relocation above — that poison the resolved env *without*
changing the weekly tag. Symptom: every cached shell shows the fragmented env
(`/usr/bin/python`, numpy ~1.23, no `lsst`) and stays that way across sessions.

Fix — force a one-time rebuild, then re-verify:

```bash
refresh-lsst-env
which python && python -c "import lsst.afw.image; print('stack OK')"
```

## Git workflow

See `CLAUDE.md` → "Git Workflow": feature branches, draft PRs, atomic commits,
never commit directly to `main`.
