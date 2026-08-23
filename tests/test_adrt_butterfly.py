"""Tests for the ADRT butterfly line-segment estimator.

Covers the closed-form inversion (`_invert_inertia`, pure numpy) and the full
`estimate_segment_adrt` on simulated top-hat segments. The estimator lives in
`adrtDetect.py`, which imports the LSST stack at module load, so these tests are
skipped when the stack (or `adrt`) is unavailable.

Derivation and validated tolerances: ``docs/detectors/adrt/butterfly.md``.
"""

import math

import numpy as np
import pytest

adrtDetect = pytest.importorskip(
    "astro_lfd.algorithms.adrtDetect",
    reason="requires the LSST stack and the adrt backend",
)
adrt = pytest.importorskip("adrt")

from astro_lfd.algorithms.adrtDetect import (  # noqa: E402
    _hesse_to_adrt,
    _invert_inertia,
    estimate_segment_adrt,
)
from astro_lfd.sims import Streak  # noqa: E402


def _rectangle_moments(length, width, phi0):
    """Central second moments of a uniform rectangle at line angle ``phi0``."""
    c, s = math.cos(phi0), math.sin(phi0)
    mu20 = (length**2 / 12.0) * c**2 + (width**2 / 12.0) * s**2
    mu02 = (length**2 / 12.0) * s**2 + (width**2 / 12.0) * c**2
    mu11 = (length**2 / 12.0 - width**2 / 12.0) * s * c
    # V(s) = A s^2 + B s + C  with (A, B, C) = (mu20, -2 mu11, mu02).
    return mu20, -2.0 * mu11, mu02


# --------------------------------------------------------------------------- #
# Pure inversion: coefficients -> (length, width, phi0). No stack needed for
# the math, but the import guard above keeps the module import consistent.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", [500.0, 1500.0, 2500.0])
@pytest.mark.parametrize("width", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("phi0_deg", [5.0, 25.0, 40.0, -30.0])
def test_invert_inertia_roundtrip(length, width, phi0_deg):
    """`_invert_inertia` exactly inverts a known rectangle inertia tensor."""
    phi0 = math.radians(phi0_deg)
    A, B, C = _rectangle_moments(length, width, phi0)

    length_hat, width_hat, phi0_hat = _invert_inertia(A, B, C)

    assert length_hat == pytest.approx(length, rel=1e-9)
    assert width_hat == pytest.approx(width, rel=1e-9)
    # atan2 orientation is modulo pi; compare via tan(phi0) (the reported slope).
    assert math.tan(phi0_hat) == pytest.approx(math.tan(phi0), abs=1e-9)


def test_invert_inertia_width_bias_subtracts_from_wsq():
    """The width-bias term is subtracted from w^2 before the root."""
    length, width, phi0 = 2000.0, 10.0, math.radians(20.0)
    A, B, C = _rectangle_moments(length, width, phi0)

    bias = 2.5
    _, width_hat, _ = _invert_inertia(A, B, C, width_bias=bias)

    assert width_hat == pytest.approx(math.sqrt(width**2 - bias), rel=1e-9)


def test_invert_inertia_clamps_negative_to_zero():
    """A width^2 driven negative by an over-large bias clamps to zero."""
    length, width, phi0 = 1000.0, 6.0, math.radians(10.0)
    A, B, C = _rectangle_moments(length, width, phi0)

    _, width_hat, _ = _invert_inertia(A, B, C, width_bias=1e6)

    assert width_hat == 0.0


# --------------------------------------------------------------------------- #
# Integration: simulate a top-hat segment, run the full estimator.
# --------------------------------------------------------------------------- #
def _run_estimator(phi0_deg, length, width, center=(2048.0, 2048.0), half_band=90):
    ny = nx = 4096
    # Streak theta is the NORMAL angle; a line at angle phi0 has normal phi0+90.
    streak = Streak.from_center_length(
        center=center, theta=phi0_deg + 90.0, length=length, peak_signal=1.0e4, width=width
    )
    signal = streak.get_signal((ny, nx), fwhm=None)
    result = adrt.adrt(signal)
    n = result.shape[2]
    q, h, s = _hesse_to_adrt(streak.rho, np.deg2rad(streak.theta), n)
    return estimate_segment_adrt(
        result, int(round(float(q))), float(h), int(round(float(s))), n, half_band=half_band
    )


@pytest.mark.parametrize("phi0_deg", [10.0, 25.0, 40.0])
@pytest.mark.parametrize("length", [1000.0, 2000.0])
@pytest.mark.parametrize("width", [8.0, 16.0])
def test_estimate_segment_recovers_geometry(phi0_deg, length, width):
    """Length and angle are near-exact; width within the discretization bias."""
    est = _run_estimator(phi0_deg, length, width)

    # Length: < 1% (validated max over the sweep is 0.68%).
    assert est.length == pytest.approx(length, rel=0.01)
    # Slope / angle: < 0.05 deg (validated max 0.006 deg).
    assert math.degrees(math.atan(est.slope)) == pytest.approx(phi0_deg, abs=0.05)
    # Width carries a small additive w^2 discretization bias; loose without it.
    assert est.width == pytest.approx(width, rel=0.15)
    # Center recovered essentially exactly.
    assert est.center_x == pytest.approx(2048.0, abs=1.0)
    assert est.center_y == pytest.approx(2048.0, abs=1.0)


def test_width_bias_is_additive_constant_in_wsq():
    """w_hat^2 - w^2 is ~constant across widths (Sheppard-type additive bias).

    Guards the width-bias calibration assumption in the derivation doc: the
    discretization error is a fixed offset in w^2, not a scale error.
    """
    phi0_deg, length = 25.0, 2000.0
    residuals = []
    for width in (4.0, 8.0, 16.0, 32.0):
        est = _run_estimator(phi0_deg, length, width)
        residuals.append(est.width**2 - width**2)

    residuals = np.array(residuals)
    # All positive, order a few px^2, and mutually consistent to ~1.5 px^2.
    assert np.all(residuals > 0)
    assert residuals.max() < 6.0
    assert residuals.std() < 1.5


def test_estimate_is_band_insensitive():
    """The globally-quadratic law gives a stable length across fit bandwidths."""
    lengths = [_run_estimator(25.0, 1500.0, 8.0, half_band=b).length for b in (60, 120, 240)]
    assert max(lengths) - min(lengths) < 0.02 * 1500.0


def test_estimate_raises_on_too_few_columns():
    """A degenerate band (fewer than three columns) is rejected."""
    est_inputs = _sim_inputs(25.0, 1500.0, 8.0)
    result, q, h, s, n = est_inputs
    with pytest.raises(ValueError, match="usable slope column"):
        # half_band=1 -> at most two interior columns around the peak.
        estimate_segment_adrt(result, q, h, s, n, half_band=1)


def _sim_inputs(phi0_deg, length, width, center=(2048.0, 2048.0)):
    ny = nx = 4096
    streak = Streak.from_center_length(
        center=center, theta=phi0_deg + 90.0, length=length, peak_signal=1.0e4, width=width
    )
    signal = streak.get_signal((ny, nx), fwhm=None)
    result = adrt.adrt(signal)
    n = result.shape[2]
    q, h, s = _hesse_to_adrt(streak.rho, np.deg2rad(streak.theta), n)
    return result, int(round(float(q))), float(h), int(round(float(s))), n
