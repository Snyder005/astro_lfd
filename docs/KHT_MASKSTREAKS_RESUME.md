# Resume context: KHTDetectTask ↔ MaskStreaksTask validation

**Purpose:** everything a future session needs to build and run the comparison
harness, without re-deriving the state of this work. Pairs with
`KHT_MASKSTREAKS_TESTPLAN.md` (what to run) and `KHT_MASKSTREAKS_DISCREPANCY.md`
(why they can differ). Written 2026-07-06.

## Where the work stands

The `TASK_ADDITION.md` work was split into three branches / PRs, all **open and
unmerged** as of writing (merge order suggested: #3 → #4 → #5):

| PR | Branch | Contents |
|---|---|---|
| #3 | `docs/package-context-and-env-caching` | `CLAUDE.md`: LSST-extension reframing + slow-first-shell note |
| #4 | `feature/kht-detect-task` | `KHTDetectTask` + tests + `knowledge/kht-detect.md` |
| #5 | `docs/kht-maskstreaks-comparison` | this file + the test plan + discrepancy report |

**Before running the harness, confirm PR #4 is merged (or its branch checked
out)** — the harness imports `astro_lfd.meas.detectStreaks`, which only exists on
that branch. This comparison branch (#5) was cut from `main` and does **not**
contain the task code.

## The one blocking decision (needs the reviewer)

`KHT_MASKSTREAKS_TESTPLAN.md` → "Inputs required" asks which input source to use.
Nothing was chosen yet. The options:

- **(A) Extend `astro_lfd.utils.testdata`** to also emit a `DETECTED` mask plane
  (threshold `streak_signal`) and optionally a trivial WCS. Deterministic, no
  external data — **recommended starting point**. Requires a small, reviewable
  addition to `src/astro_lfd/utils/testdata.py` (currently its mask plane is all
  zeros — see `simulate_exposure`, `mask = np.zeros(...)`).
- **(B) A real calexp / difference image** with `DETECTED`/`SAT` populated and a
  known streak. Exercises the SAT-dilation + bad-plane paths (A) can't. Needs a
  file path / dataId + which planes are set from the reviewer.

**Action on resume:** if unanswered, ask the user (A/B/both) before coding the
harness. If (A) is chosen, the first code step is the testdata `DETECTED`-plane
addition, on a new branch off merged `main`.

## Key facts to not re-derive

These are already established from reading both sources; trust them:

- **Shared engine:** both tasks use the same Canny params, the same
  `lsst.kht.find_lines` call, the same recursive-KMeans stop rule, and the
  **same `LineProfile` fit** (`KHTDetectTask` imports `Line`/`LineProfile` from
  `lsst.meas.algorithms.maskStreaks`). The fit is therefore **not** a discrepancy
  source — differences are pre-fit (edges/weights) and post-fit (frame/canon).
- **Entry points:** reference is `MaskStreaksTask().find(maskedImage)` →
  `.lines` (a `LineCollection` of center-relative `Line(rho, theta[deg], …)`).
  New task is `KHTDetectTask().run(table, exposure)` → `.streaks` (a
  `SourceCatalog`; wrap each record in `StreakAdapter` and call `.getLine()`).
  Build `table` from `StreakAdapter.makeMinimalSchema()`.
- **Frame reconciliation (the crux, discrepancy §1):** reference lines are
  image-**center**-relative; `KHTDetectTask` translates by `bbox.getCenter()`
  into absolute pixels and canonicalizes `theta → [0, π)` (flipping `rho` sign,
  discrepancy §2). To compare, map **both** to one frame + one canonicalization
  before matching. Mind `XY0 != (0,0)` for sub-images/calexps.
- **Matcher already available:** `astro_lfd.geom.line.embed_rho_theta(rho, theta,
  rho_tol, theta_tol)` gives a tolerance-scaled Euclidean embedding for
  nearest-neighbor pairing; tie tolerances to `rho_bin_size` / `theta_bin_size`.
- **Determinism control (discrepancy §4):** KMeans has no fixed seed in either
  task → run ≥3× to establish the jitter floor before calling a delta "real".
- **Config alignment:** set both configs to identical shared values, and note
  `KHTDetectTask.bad_mask_planes` adds `ITL_DIP`, `SPIKE` vs. the reference set —
  temporarily drop them to isolate discrepancy §3.

## Environment reminders

- Verify the stack first: `which python` → `.../lsst-scipipe-*/bin/python`. The
  **first Bash call of a session is slow** (NFS stack resolution, can time out) —
  let it finish once; it's cached after. See `CONTRIBUTING.md` → "Speeding up the
  LSST stack" and `refresh-lsst-env`.
- Both `lsst.kht` and `lsst.meas.algorithms.maskStreaks` import cleanly on the
  SDF stack (`lsst-scipipe-13.0.0`, verified 2026-07-06).
- Git push uses the HTTPS + `gh`-token workaround (no SSH/MFA).

## Deliverable on completion

Append a results table (counts, matched/unmatched, `Δrho`/`Δtheta`/`Δlength`
distributions, KMeans jitter floor) to `KHT_MASKSTREAKS_DISCREPANCY.md`,
confirming or re-ranking its hypotheses. Do this on a fresh branch (e.g.
`feature/kht-maskstreaks-validation`) once inputs are approved.
