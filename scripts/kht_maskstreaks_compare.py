"""Empirical comparison harness: KHTDetectTask vs. MaskStreaksTask.

Runs both streak detectors on the *same* input exposure, maps their outputs
into one common frame + canonicalization, matches lines by (rho, theta), and
reports count agreement, per-match residuals, and the KMeans jitter floor.

This is a validation *script*, not library code -- it lives under ``scripts/``
and is not imported by the package.  It backs the report appended to
``docs/detectors/kht/maskstreaks-discrepancy.md``.

Two input sources (see ``docs/detectors/kht/maskstreaks-testplan.md``):

* ``--source testdata`` (Option A): a synthetic exposure from
  ``astro_lfd.utils.testdata`` with a ``DETECTED`` plane thresholded from the
  noise-free streak signal.  Deterministic, no external data.
* ``--source butler`` (Option B): a real ``difference_image`` fetched from the
  Butler repo in ``notebooks/Example_Visit.ipynb``.

Both tasks share the same detection engine (Canny -> ``lsst.kht`` -> recursive
KMeans -> Moffat ``LineProfile`` fit); ``KHTDetectTask`` imports ``Line`` /
``LineProfile`` from ``maskStreaks``.  The harness does not modify task code.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import lsst.afw.table as afwTable
import lsst.geom as geom

from astro_lfd.geom.line import Line2D, embed_rho_theta
from astro_lfd.meas.detectStreaks import KHTDetectTask, KHTDetectConfig
from astro_lfd.table.streakAdapter import StreakAdapter
from astro_lfd.utils import testdata

# The reference implementation and the new task.
from lsst.meas.algorithms.maskStreaks import MaskStreaksTask, MaskStreaksConfig

# Reference bad-plane set (drop the KHT additions ITL_DIP, SPIKE to align #3).
REFERENCE_BAD_PLANES = ["NO_DATA", "INTRP", "BAD", "SAT", "EDGE"]


@dataclass
class CommonLine:
    """A line canonicalized into the absolute-pixel, ``theta in [0, pi)`` frame."""

    rho: float
    theta: float  # radians, in [0, pi)
    length: float | None  # segment length in pixels (None for infinite ref line)
    source: str


# --------------------------------------------------------------------------
# KMeans determinism control
# --------------------------------------------------------------------------
@contextlib.contextmanager
def fixed_kmeans_seed(random_state: int | None):
    """Force a fixed ``random_state`` into both tasks' ``KMeans`` calls.

    Both ``KHTDetectTask._cluster_lines`` and ``MaskStreaksTask._findClusters``
    construct ``KMeans(n_clusters=..., n_init="auto")`` with no seed.  To get a
    matched, jitter-free comparison we monkeypatch the ``KMeans`` symbol each
    module imported so it injects ``random_state``.  With ``random_state=None``
    this is a no-op (native behaviour).
    """
    if random_state is None:
        yield
        return

    import astro_lfd.meas.detectStreaks as kht_mod
    import lsst.meas.algorithms.maskStreaks as ms_mod

    originals = {}
    for mod in (kht_mod, ms_mod):
        original = mod.KMeans

        def make_seeded(orig):
            def seeded(*args, **kwargs):
                kwargs.setdefault("random_state", random_state)
                return orig(*args, **kwargs)

            return seeded

        originals[mod] = original
        mod.KMeans = make_seeded(original)
    try:
        yield
    finally:
        for mod, original in originals.items():
            mod.KMeans = original


# --------------------------------------------------------------------------
# Canonicalization: map both outputs into one frame
# --------------------------------------------------------------------------
def ref_line_to_common(line, box: geom.Box2I) -> CommonLine:
    """Map a reference ``maskStreaks.Line`` into the common frame.

    ``MaskStreaksTask`` returns image-**center**-relative ``(rho, theta[deg])``.
    Wrapping it in ``Line2D`` and translating by ``box.getCenter()`` applies
    exactly the same transform (translate + ``_canonicalize`` to ``[0, pi)``)
    that ``KHTDetectTask`` applies to its own fit, so both land in the same
    absolute-pixel, canonical frame.
    """
    shift = geom.Extent2D(box.getCenter())
    line2d = Line2D(line.rho, line.theta * geom.degrees).translated(shift)
    segment = line2d.intersection(box)
    length = segment.length if segment is not None else None
    return CommonLine(rho=line2d.rho, theta=line2d.theta.asRadians(), length=length, source="maskStreaks")


def kht_record_to_common(record) -> CommonLine:
    """Map a ``KHTDetectTask`` catalog record into the common frame.

    ``KHTDetectTask`` already stores its line in the absolute-pixel canonical
    frame via ``StreakAdapter`` / ``Line2D``.
    """
    adapter = StreakAdapter(record)
    segment = adapter.getLineSegment()
    return CommonLine(
        rho=segment.rho,
        theta=segment.theta.asRadians(),
        length=segment.length,
        source="kht",
    )


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def match_lines(
    ref: list[CommonLine],
    test: list[CommonLine],
    rho_tol: float,
    theta_tol: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy nearest-neighbour match in the tolerance-scaled embedding.

    Returns ``(pairs, unmatched_ref, unmatched_test)`` where each pair is
    ``(ref_index, test_index, embedded_distance)``.  A pair is accepted only if
    its embedded distance is <= 1 (i.e. within the (rho_tol, theta_tol) box).
    """
    if not ref or not test:
        return [], list(range(len(ref))), list(range(len(test)))

    theta_tol_rad = np.deg2rad(theta_tol)
    ref_emb = embed_rho_theta(
        [c.rho for c in ref], [c.theta for c in ref], rho_tol, theta_tol_rad
    )
    test_emb = embed_rho_theta(
        [c.rho for c in test], [c.theta for c in test], rho_tol, theta_tol_rad
    )

    # Pairwise embedded distances.
    dists = np.linalg.norm(ref_emb[:, None, :] - test_emb[None, :, :], axis=2)

    pairs: list[tuple[int, int, float]] = []
    used_ref: set[int] = set()
    used_test: set[int] = set()
    # Greedy: repeatedly take the globally-closest available pair within tol.
    order = np.dstack(np.unravel_index(np.argsort(dists, axis=None), dists.shape))[0]
    for i, j in order:
        i, j = int(i), int(j)
        if i in used_ref or j in used_test:
            continue
        if dists[i, j] > 1.0:
            break
        pairs.append((i, j, float(dists[i, j])))
        used_ref.add(i)
        used_test.add(j)

    unmatched_ref = [i for i in range(len(ref)) if i not in used_ref]
    unmatched_test = [j for j in range(len(test)) if j not in used_test]
    return pairs, unmatched_ref, unmatched_test


def theta_delta_deg(a: float, b: float) -> float:
    """Absolute angular difference (deg) between two [0, pi) normal angles."""
    d = abs(a - b) % np.pi
    return np.rad2deg(min(d, np.pi - d))


# --------------------------------------------------------------------------
# Task runners
# --------------------------------------------------------------------------
def make_configs() -> tuple[KHTDetectConfig, MaskStreaksConfig]:
    """Build aligned configs for the two tasks (shared values + bad planes)."""
    kht_config = KHTDetectConfig()
    kht_config.bad_mask_planes = REFERENCE_BAD_PLANES  # drop ITL_DIP, SPIKE (#3)

    ms_config = MaskStreaksConfig()
    ms_config.badMaskPlanes = REFERENCE_BAD_PLANES
    # All shared numeric knobs already share defaults; keep them explicit-equal.
    ms_config.rhoBinSize = kht_config.rho_bin_size
    ms_config.thetaBinSize = kht_config.theta_bin_size
    ms_config.invSigma = kht_config.inv_sigma
    ms_config.footprintThreshold = kht_config.footprint_threshold
    ms_config.dChi2Tolerance = kht_config.dchi2_tolerance
    ms_config.maxFitIter = kht_config.max_fit_iter
    ms_config.nSigmaMask = kht_config.nsigma_mask
    ms_config.saturatedDetectionsDilation = kht_config.saturated_detections_dilation
    return kht_config, ms_config


def run_reference(exposure, ms_config: MaskStreaksConfig) -> list:
    """Run ``MaskStreaksTask.find`` and return its ``lines`` (raw ``Line``s)."""
    task = MaskStreaksTask(config=ms_config)
    out = task.find(exposure.maskedImage)
    return list(out.lines)


def run_kht(exposure, kht_config: KHTDetectConfig):
    """Run ``KHTDetectTask.run`` and return its streak ``SourceCatalog``."""
    schema = StreakAdapter.makeMinimalSchema()
    table = afwTable.SourceTable.make(schema)
    task = KHTDetectTask(config=kht_config)
    out = task.run(table, exposure)
    return out.streaks


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def make_testdata_exposure(seed: int, detected_nsigma: float = 5.0):
    """Option A: synthetic exposure with a DETECTED plane on the streak ridge.

    The DETECTED plane is thresholded from the *noise-free* streak signal at
    ``detected_nsigma`` * per-pixel sigma, so it deterministically flags the
    ridge without modifying ``testdata.py``.
    """
    streak = testdata.StreakConfig(theta=30.0, rho=1500.0, width=20.0, peak_signal=1500.0)
    ti = testdata.simulate_exposure(streak, band="i", seed=seed)

    sigma = (testdata.FWHM / testdata.PIXEL_SCALE) * testdata.FWHM_TO_SIGMA
    signal = testdata.streak_signal(ti.image.shape, streak, sigma)  # electrons
    noise = np.sqrt(testdata.READ_NOISE**2 + testdata.SKY_COUNTS["i"] + signal)
    detected = signal > (detected_nsigma * noise)

    exposure = testdata.to_exposure(ti)
    exposure.mask.array[detected] |= exposure.mask.getPlaneBitMask("DETECTED")
    return exposure


def make_butler_exposure(visit: int, detector: int):
    """Option B: fetch a real difference image from the Butler."""
    from lsst.daf.butler import Butler
    from lsst.obs.lsst import LsstCam

    repo = "/sdf/data/rubin/repo/dp2_prep"
    collections = "u/snyder18/DRP/detectAndMeasureDiaSource"
    instrument = LsstCam().getCamera().getName()
    butler = Butler(repo, collections=collections, instrument=instrument)
    return butler.get(
        "difference_image", instrument=instrument, visit=visit, detector=detector
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def summarize(label: str, ref: list[CommonLine], test: list[CommonLine],
              rho_tol: float, theta_tol: float) -> dict:
    """Match and print a residual summary for one paired run."""
    pairs, un_ref, un_test = match_lines(ref, test, rho_tol, theta_tol)
    print(f"\n=== {label} ===")
    print(f"  reference lines: {len(ref)}   kht lines: {len(test)}   matched: {len(pairs)}")
    print(f"  unmatched reference: {len(un_ref)}   unmatched kht: {len(un_test)}")
    d_rho, d_theta, d_len = [], [], []
    for i, j, dist in pairs:
        dr = abs(ref[i].rho - test[j].rho)
        dt = theta_delta_deg(ref[i].theta, test[j].theta)
        d_rho.append(dr)
        d_theta.append(dt)
        if ref[i].length is not None and test[j].length is not None:
            d_len.append(abs(ref[i].length - test[j].length))
        print(f"    pair ref#{i}<->kht#{j}: "
              f"rho {ref[i].rho:9.3f} vs {test[j].rho:9.3f} (drho={dr:7.4f})  "
              f"theta {np.rad2deg(ref[i].theta):7.3f} vs {np.rad2deg(test[j].theta):7.3f} deg "
              f"(dtheta={dt:.4f})")
    if pairs:
        print(f"  |drho|   max={max(d_rho):.4f}  mean={np.mean(d_rho):.4f} px")
        print(f"  |dtheta| max={max(d_theta):.4f}  mean={np.mean(d_theta):.4f} deg")
        if d_len:
            print(f"  |dlen|   max={max(d_len):.4f}  mean={np.mean(d_len):.4f} px")
    return {
        "n_ref": len(ref), "n_test": len(test), "n_matched": len(pairs),
        "n_unmatched_ref": len(un_ref), "n_unmatched_test": len(un_test),
        "d_rho": d_rho, "d_theta": d_theta, "d_len": d_len,
    }


def jitter_floor(exposure, kht_config, ms_config, n_runs: int, rho_tol, theta_tol):
    """Run each task ``n_runs`` times unseeded; report within-task spread.

    This is the KMeans random-init floor (#4): any KHT-vs-reference delta below
    this is jitter, not a code difference.
    """
    box = exposure.getBBox()
    print(f"\n########## JITTER FLOOR ({n_runs} unseeded runs each) ##########")

    def collect(runner, to_common):
        runs = []
        for _ in range(n_runs):
            out = runner()
            runs.append([to_common(x) for x in out])
        return runs

    ref_runs = collect(lambda: run_reference(exposure, ms_config),
                        lambda l: ref_line_to_common(l, box))
    kht_runs = collect(lambda: run_kht(exposure, kht_config),
                       kht_record_to_common)

    print(f"  reference line counts across runs: {[len(r) for r in ref_runs]}")
    print(f"  kht       line counts across runs: {[len(r) for r in kht_runs]}")

    def self_spread(runs, name):
        base = runs[0]
        max_dr, max_dt = 0.0, 0.0
        for other in runs[1:]:
            pairs, _, _ = match_lines(base, other, rho_tol, theta_tol)
            for i, j, _ in pairs:
                max_dr = max(max_dr, abs(base[i].rho - other[j].rho))
                max_dt = max(max_dt, theta_delta_deg(base[i].theta, other[j].theta))
        print(f"  {name} self-jitter: max|drho|={max_dr:.4f} px  max|dtheta|={max_dt:.4f} deg")
        return max_dr, max_dt

    self_spread(ref_runs, "reference")
    self_spread(kht_runs, "kht      ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["testdata", "butler"], default="testdata")
    parser.add_argument("--seed", type=int, default=12345, help="testdata sim seed")
    parser.add_argument("--visit", type=int, default=2025071700631)
    parser.add_argument("--detector", type=int, default=140)
    parser.add_argument("--random-state", type=int, default=0,
                        help="fixed KMeans seed for the matched run (use -1 for None)")
    parser.add_argument("--jitter-runs", type=int, default=3)
    args = parser.parse_args()

    kht_config, ms_config = make_configs()
    random_state = None if args.random_state < 0 else args.random_state

    if args.source == "testdata":
        exposure = make_testdata_exposure(args.seed)
        label = f"testdata (seed={args.seed})"
    else:
        exposure = make_butler_exposure(args.visit, args.detector)
        label = f"butler visit={args.visit} det={args.detector}"

    box = exposure.getBBox()
    n_detected = int((exposure.mask.array & exposure.mask.getPlaneBitMask("DETECTED") != 0).sum())
    print(f"Input: {label}")
    print(f"  bbox={box}  XY0={exposure.getXY0()}  DETECTED px={n_detected}  "
          f"hasWcs={exposure.getWcs() is not None}")

    rho_tol = kht_config.rho_bin_size
    theta_tol = kht_config.theta_bin_size

    # Matched, seeded run (removes #4 jitter).
    with fixed_kmeans_seed(random_state):
        ref_lines = run_reference(exposure, ms_config)
        kht_streaks = run_kht(exposure, kht_config)
    ref = [ref_line_to_common(l, box) for l in ref_lines]
    test = [kht_record_to_common(r) for r in kht_streaks]
    summarize(f"MATCHED RUN (random_state={random_state})", ref, test, rho_tol, theta_tol)

    # Jitter floor (unseeded).
    jitter_floor(exposure, kht_config, ms_config, args.jitter_runs, rho_tol, theta_tol)


if __name__ == "__main__":
    main()
