# Astro LFD

This repository contains `astro_lfd`, a Python library for Linear Feature
Detection in astronomical images acquired by the LSST Camera. This document
contains critical information about working with this codebase. Follow these
guidelines precisely.

## Convention Hierarchy

When sources conflict, follow this precedence (higher overrides lower):

| Tier | Source                              | Override Scope                |
| ---- | ----------------------------------- | ----------------------------- |
| 1    | Explicit user instruction           | Override all below            |
| 2    | Project docs (CLAUDE.md, README.md) | Override conventions/defaults |
| 3    | Universal best practices            | Confirm if uncertain          |

**Conflict resolution**: Lower tier numbers win. Subdirectory docs override root docs for that subtree.

## Knowledge Strategy

**CLAUDE.md** = navigation index (WHAT is here, WHEN to read)
**README.md** = invisible knowledge (WHY it's structured this way)

## Core Workflow

All tasks should be performed within the scope of the Git Workflow. The generalized pattern is

1. Parse task; retrieve additional context only for determining the scope of the task.
2. Stage within git workflow (checkout branch, link issues).
2. Develop implementation plan.
3. Implement 
4. Evaluate
5. Document
6. Finalize within git workflow (add/commit changes, create pull request)

### Git Workflow

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

## Environment & Dependencies

`astro_lfd` is developed against the shared Rubin/LSST stack on SDF. The shared
LSST stack under `/sdf/group/rubin/sw/` is **read-only** and updates weekly;
you are a stack *user* and cannot add packages to it. So the working 
environment is built in two orthogonal layers on top of a **clean login
baseline** (`.bashrc` sets your profile, **no conda env activated, no stack
sourced**):

### Setup — a Claude Code session (via the cache loader)

A Claude session starts from the clean baseline but **must not** re-source
`loadLSST.sh` on every Bash call (slow over NFS, times out). Instead it relies 
on the cache loader wired via `BASH_ENV`.

The per-session procedure is:
1. The **first** Bash command in a session resolves `loadLSST.sh` +
`setup lsst_distrib` over NFS, which takes tens of seconds and can time out. 
Don't interpret an early hang as a broken command — let it finish (or re-run) 
once and it is cached.
2. A cache-loader snapshots the resolved environment so every subsequent call
is fast, and self-heals when the `w_latest` target changes.

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

The `astro_lfd` editable install (user-site, step 3 above) persists across 
cached shells, so a session does **not** reinstall; it only needs the stack to 
be active. If `import astro_lfd` fails while the stack *is* active, it simply 
hasn't been installed yet; ask the user before running a `--user` install on 
their behalf.

Full recipe and gotchas are in **`CONTRIBUTING.md`**, but should only be parsed
if failures occur.

## Repository Layout
