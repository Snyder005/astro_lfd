"""Tests for multi-peak detection in `ADRTDetectTask._find_peaks`.

Simulated streaks (`astro_lfd.sims.Streak`) with unit white noise are pushed
through ``adrt.adrt`` and the peak detector; every test asserts the exact
number of recovered lines and matches each to its truth in the PIXEL frame.
The simulation helpers import the LSST stack, so these tests are skipped
when the stack (or `adrt`) is unavailable.

Design notes and tuning rationale: ``docs/detectors/adrt/peak-detection-plan.md``.
"""

import math
import time

import numpy as np
import pytest

pytest.importorskip("lsst.geom", reason="requires the LSST stack")
adrt = pytest.importorskip("adrt")

from astro_lfd.algorithms.adrtDetect import ADRTDetectTask, AdrtPeak  # noqa: E402
from astro_lfd.geom import Line2D  # noqa: E402
from astro_lfd.sims import Streak  # noqa: E402

N = 1024
FWHM = 3.0
WIDTH = 2.0
RHO_TOL = 3.0  # pixels
THETA_TOL = 0.3  # degrees


def _make_image(streaks, seed, shape=(N, N)):
    """Sum the streak signals and add unit white noise."""
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 1.0, shape)
    for streak in streaks:
        image += streak.get_signal(shape, fwhm=FWHM)
    return image.astype(np.float32)


def _streak(center, theta, length, peak):
    return Streak.from_center_length(center, theta, length, peak, WIDTH)


def _line_distance(line, streak):
    """(|d rho|, |d theta| in degrees) between a `Line2D` and a truth `Streak`."""
    rho_a, theta_a = line.rho, line.theta.asDegrees()
    rho_b, theta_b = streak.rho, streak.theta
    dtheta = abs(theta_a - theta_b)
    if dtheta > 90.0:
        return abs(rho_a + rho_b), 180.0 - dtheta
    return abs(rho_a - rho_b), dtheta


def _assert_matches(lines, streaks):
    """Every truth streak is matched by exactly one line and no line is spurious."""
    assert len(lines) == len(streaks), [(round(ln.rho, 1), round(ln.theta.asDegrees(), 2)) for ln in lines]
    unmatched = list(streaks)
    for line in lines:
        hits = [s for s in unmatched if _line_distance(line, s) <= (RHO_TOL, THETA_TOL)]
        assert hits, f"spurious line rho={line.rho:.1f} theta={line.theta.asDegrees():.2f}"
        unmatched.remove(hits[0])
    assert not unmatched


@pytest.fixture(scope="module")
def task():
    return ADRTDetectTask()


def _detect(task, streaks, seed, **kwargs):
    return task._find_peaks(adrt.adrt(_make_image(streaks, seed)), **kwargs)


# ---------------------------------------------------------------------------
# Scene tests: exact counts and geometry
# ---------------------------------------------------------------------------

SCENES = {
    "bright_single": [_streak((512.0, 520.0), 115.0, 700.0, 5.0)],
    "very_bright": [_streak((500.0, 500.0), 60.0, 900.0, 50.0)],
    "bright_faint_cross": [
        _streak((512.0, 512.0), 115.0, 800.0, 10.0),
        _streak((520.0, 500.0), 30.0, 800.0, 2.0),
    ],
    "faint_parallel_pair": [
        _streak((512.0, 512.0), 70.0, 800.0, 2.0),
        _streak((512.0, 560.0), 71.0, 800.0, 2.0),
    ],
    "three_mixed": [
        _streak((512.0, 512.0), 115.0, 700.0, 5.0),
        _streak((600.0, 400.0), 30.0, 600.0, 2.0),
        _streak((300.0, 700.0), 178.0, 500.0, 10.0),
    ],
    "short_bright": [_streak((400.0, 600.0), 140.0, 150.0, 20.0)],
    "seam_45": [_streak((512.0, 512.0), 45.0, 800.0, 5.0)],
    "seam_90": [_streak((512.0, 512.0), 90.0, 800.0, 5.0)],
    "seam_0": [_streak((512.0, 512.0), 179.5, 800.0, 5.0)],
    "near_seam_bright": [_streak((512.0, 512.0), 44.0, 800.0, 30.0)],
    "very_faint": [_streak((512.0, 512.0), 115.0, 900.0, 0.5)],
}


@pytest.mark.parametrize("name", sorted(SCENES))
@pytest.mark.parametrize("seed", [0, 1])
def test_scene_recovers_all_streaks(task, name, seed):
    lines = _detect(task, SCENES[name], seed)
    _assert_matches(lines, SCENES[name])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pure_noise_yields_nothing(task, seed):
    assert _detect(task, [], seed) == []


def test_lines_are_pixel_frame_line2d(task):
    lines = _detect(task, SCENES["bright_single"], 0)
    assert all(isinstance(ln, Line2D) for ln in lines)
    assert 0.0 <= lines[0].theta.asRadians() < math.pi


def test_ranked_by_significance(task):
    lines, peaks = _detect(task, SCENES["three_mixed"], 0, return_peaks=True)
    values = [p.value for p in peaks]
    assert values == sorted(values, reverse=True)
    # The brightest (10 sigma/px) streak comes first.
    assert _line_distance(lines[0], SCENES["three_mixed"][2]) <= (RHO_TOL, THETA_TOL)


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def test_max_peaks_truncates(task):
    lines = _detect(task, SCENES["three_mixed"], 0, max_peaks=1)
    assert len(lines) == 1


def test_return_peaks_descriptors(task):
    lines, peaks = _detect(task, SCENES["bright_single"], 0, return_peaks=True)
    assert len(lines) == len(peaks) == 1
    peak = peaks[0]
    assert isinstance(peak, AdrtPeak)
    assert peak.line is lines[0]
    assert 0 <= peak.q < 4 and 0 <= peak.h < 2 * N - 1 and 0 <= peak.s < N
    assert abs(peak.h_refined - peak.h) <= 0.5 and abs(peak.s_refined - peak.s) <= 0.5


@pytest.mark.parametrize("refine", ["none", "parabolic"])
def test_refine_modes(task, refine):
    lines = _detect(task, SCENES["bright_single"], 0, refine=refine)
    _assert_matches(lines, SCENES["bright_single"])


def test_refine_none_is_on_grid(task):
    _, peaks = _detect(task, SCENES["bright_single"], 0, refine="none", return_peaks=True)
    assert peaks[0].h_refined == peaks[0].h and peaks[0].s_refined == peaks[0].s


def test_refine_butterfly_runs(task):
    # Butterfly moments are noise-dominated at unit per-pixel noise, so only
    # the count and the return type are checked here.
    lines = _detect(task, SCENES["very_bright"], 0, refine="butterfly")
    assert len(lines) == 1 and isinstance(lines[0], Line2D)


def test_explicit_line_length(task):
    image = _make_image(SCENES["bright_single"], 0)
    line_length = adrt.adrt(np.ones_like(image))
    lines = task._find_peaks(adrt.adrt(image), line_length=line_length)
    _assert_matches(lines, SCENES["bright_single"])


def test_invalid_refine_raises(task):
    with pytest.raises(ValueError, match="refinement"):
        task._find_peaks(np.zeros((4, 2 * N - 1, N), dtype=np.float32), refine="cubic")


@pytest.mark.parametrize("shape", [(4, N, N), (3, 2 * N - 1, N), (2 * N - 1, N)])
def test_invalid_shape_raises(task, shape):
    with pytest.raises(ValueError, match="shape"):
        task._find_peaks(np.zeros(shape, dtype=np.float32))


def test_detect_returns_lines(task):
    image = _make_image(SCENES["bright_single"], 0)
    lines = task.detect(image)
    _assert_matches(lines, SCENES["bright_single"])


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_find_peaks_timing_2048(task):
    shape = (2048, 2048)
    streaks = [_streak((1024.0, 1024.0), 115.0, 1500.0, 5.0)]
    accumulator = adrt.adrt(_make_image(streaks, 0, shape))
    task._find_peaks(accumulator)  # warm the line-length cache
    start = time.perf_counter()
    lines = task._find_peaks(accumulator)
    elapsed = time.perf_counter() - start
    _assert_matches(lines, streaks)
    assert elapsed < 10.0, f"_find_peaks took {elapsed:.1f} s at N=2048"
