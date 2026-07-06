"""Unit tests for :mod:`astro_adrt.testdata`.

Covers the streak profile, the radiometric simulation (shapes/dtypes, unit
conversion, reproducibility), and both I/O paths.  The afw FITS tests are
skipped when ``lsst.afw.image`` is unavailable so the core suite runs without
the LSST stack.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astro_adrt import testdata as td

# A small, fast, power-of-two-square shape so the ADRT could consume the output
# directly (the detector requires square power-of-two float input).
SHAPE = (256, 256)


# --- blurred_box ------------------------------------------------------------
def test_blurred_box_peak_normalized() -> None:
    """Profile peaks at 1.0 at the ridge center and is symmetric."""
    x = np.linspace(-100, 100, 1001)
    y = td.blurred_box(x, width=20.0, sigma=3.0)
    assert y.max() == pytest.approx(1.0, abs=1e-6)
    assert y[x == 0.0] == pytest.approx(1.0, abs=1e-6)
    # Symmetric about x = 0.
    assert np.allclose(y, y[::-1], atol=1e-9)


def test_blurred_box_decays_off_line() -> None:
    """Signal far from the line is ~0; inside the box it is near the peak."""
    assert td.blurred_box(np.array([1000.0]), 20.0, 3.0)[0] == pytest.approx(0.0, abs=1e-9)
    assert td.blurred_box(np.array([0.0]), 20.0, 3.0)[0] > 0.99


# --- streak_signal ----------------------------------------------------------
def test_streak_signal_shape_and_nonnegative() -> None:
    """Signal matches image shape, is finite, and non-negative."""
    sc = td.StreakConfig(theta=0.0, rho=128.0, width=20.0, peak_signal=1000.0)
    sig = td.streak_signal(SHAPE, sc, sigma=3.0)
    assert sig.shape == SHAPE
    assert np.isfinite(sig).all()
    assert (sig >= 0.0).all()


def test_streak_signal_locates_vertical_line() -> None:
    """theta=0 places the ridge at column x = rho (vertical line)."""
    rho = 100.0
    sc = td.StreakConfig(theta=0.0, rho=rho, width=20.0, peak_signal=1000.0)
    sig = td.streak_signal(SHAPE, sc, sigma=3.0)
    # Row-summed signal peaks at the column nearest rho.
    assert int(np.argmax(sig.sum(axis=0))) == round(rho)


def test_streak_signal_uses_degrees() -> None:
    """theta is interpreted in degrees, not radians (regression test).

    A radians bug would make theta=90 nearly identical to theta=0; in degrees
    the ridge is horizontal (constant along rows) vs. vertical.
    """
    rho = 100.0
    vertical = td.streak_signal(SHAPE, td.StreakConfig(theta=0.0, rho=rho), sigma=3.0)
    horizontal = td.streak_signal(SHAPE, td.StreakConfig(theta=90.0, rho=rho), sigma=3.0)
    # Vertical: ridge runs down a column -> peak in the column profile.
    assert int(np.argmax(vertical.sum(axis=0))) == round(rho)
    # Horizontal: ridge runs across a row -> peak in the row profile.
    assert int(np.argmax(horizontal.sum(axis=1))) == round(rho)


# --- simulate_exposure ------------------------------------------------------
def test_simulate_exposure_shapes_and_dtypes() -> None:
    """Image/variance are float32, mask is int32, all match the requested shape."""
    ti = td.simulate_exposure(td.StreakConfig(), shape=SHAPE, seed=0)
    for plane in (ti.image, ti.variance, ti.mask):
        assert plane.shape == SHAPE
    assert ti.image.dtype == np.float32
    assert ti.variance.dtype == np.float32
    assert ti.mask.dtype == np.int32


def test_simulate_exposure_mask_empty_and_variance_positive() -> None:
    """Mask starts all-good (0) and variance is strictly positive everywhere."""
    ti = td.simulate_exposure(td.StreakConfig(), shape=SHAPE, seed=0)
    assert int(ti.mask.max()) == 0
    assert (ti.variance > 0).all()


def test_simulate_exposure_background_subtracted() -> None:
    """Off-streak background is ~0 (sky mean removed)."""
    # Place the streak off the frame so the whole image is background.
    sc = td.StreakConfig(theta=0.0, rho=10_000.0, width=20.0, peak_signal=1000.0)
    ti = td.simulate_exposure(sc, band="i", shape=SHAPE, calib=1.0, unit="electron", seed=1)
    # Mean within a few sigma of 0 given read noise + sky Poisson over N pixels.
    noise_per_pixel = math.sqrt(td.READ_NOISE**2 + td.SKY_COUNTS["i"])
    sem = noise_per_pixel / math.sqrt(ti.image.size)
    assert abs(float(ti.image.mean())) < 5 * sem


def test_simulate_exposure_reproducible_with_seed() -> None:
    """Same seed -> identical arrays; different seed -> different arrays."""
    a = td.simulate_exposure(td.StreakConfig(), shape=SHAPE, seed=42)
    b = td.simulate_exposure(td.StreakConfig(), shape=SHAPE, seed=42)
    c = td.simulate_exposure(td.StreakConfig(), shape=SHAPE, seed=43)
    assert np.array_equal(a.image, b.image)
    assert not np.array_equal(a.image, c.image)


def test_simulate_exposure_calib_scaling() -> None:
    """calib divides the image and its square divides the variance."""
    common = dict(shape=SHAPE, band="i", seed=7)
    electrons = td.simulate_exposure(td.StreakConfig(), calib=1.0, unit="electron", **common)
    scaled = td.simulate_exposure(td.StreakConfig(), calib=2.0, unit="nJy", **common)
    # Same seed -> same electron draw, so the conversion is exact.
    assert np.allclose(scaled.image, electrons.image / 2.0, rtol=1e-5)
    assert np.allclose(scaled.variance, electrons.variance / 4.0, rtol=1e-5)


def test_simulate_exposure_metadata() -> None:
    """Provenance is recorded, including streak params and the unit label."""
    sc = td.StreakConfig(theta=30.0, rho=128.0, width=15.0, peak_signal=500.0)
    ti = td.simulate_exposure(sc, band="g", calib=1.5, unit="nJy", shape=SHAPE, seed=3)
    assert ti.meta["BUNIT"] == "nJy"
    assert ti.meta["CALIB"] == 1.5
    assert ti.meta["BAND"] == "g"
    assert ti.meta["STRK_THETA"] == 30.0
    assert ti.meta["STRK_RHO"] == 128.0


def test_simulate_exposure_bad_band_raises() -> None:
    """An unknown band is a KeyError from the sky-level lookup."""
    with pytest.raises(KeyError):
        td.simulate_exposure(td.StreakConfig(), band="x", shape=SHAPE, seed=0)


# --- npz I/O ----------------------------------------------------------------
def test_npz_roundtrip(tmp_path) -> None:
    """save_npz -> load_npz preserves arrays and metadata exactly."""
    ti = td.simulate_exposure(td.StreakConfig(theta=10.0), shape=SHAPE, seed=5)
    path = str(tmp_path / "img.npz")
    td.save_npz(ti, path)
    out = td.load_npz(path)
    assert np.array_equal(out.image, ti.image)
    assert np.array_equal(out.variance, ti.variance)
    assert np.array_equal(out.mask, ti.mask)
    assert out.meta == ti.meta


# --- afw FITS I/O (optional) ------------------------------------------------
afw = pytest.importorskip("lsst.afw.image", reason="LSST afw not available")


def test_fits_roundtrip(tmp_path) -> None:
    """save_fits -> ExposureF read back preserves planes; header carries BUNIT."""
    ti = td.simulate_exposure(td.StreakConfig(theta=45.0), shape=SHAPE, unit="nJy", seed=9)
    path = str(tmp_path / "img.fits")
    td.save_fits(ti, path)
    exp = afw.ExposureF(path)
    assert np.allclose(exp.image.array, ti.image)
    assert np.allclose(exp.variance.array, ti.variance)
    assert np.array_equal(exp.mask.array, ti.mask)
    assert exp.getMetadata().get("BUNIT") == "nJy"


def test_from_exposure_roundtrip() -> None:
    """to_exposure -> from_exposure preserves arrays."""
    ti = td.simulate_exposure(td.StreakConfig(), shape=SHAPE, seed=11)
    out = td.from_exposure(td.to_exposure(ti))
    assert np.allclose(out.image, ti.image)
    assert np.allclose(out.variance, ti.variance)
    assert np.array_equal(out.mask, ti.mask)
