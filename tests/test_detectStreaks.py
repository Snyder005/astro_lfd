"""Unit tests for :mod:`astro_lfd.meas.detectStreaks`.

Covers the pure/near-pure helpers (`get_pixel_mask`, `binary_dilation`,
`_cluster_lines`), the config defaults, and task construction.  These do **not**
run the full task -- exercising the KHT + profile fit end to end needs a
realistic exposure and is deferred to the comparison test plan.  The module is
skipped when the LSST measurement stack is unavailable so the core suite still
runs without it.
"""

from __future__ import annotations

import numpy as np
import pytest

# The task imports the maskStreaks profile fitter, afw, and lsst.kht, so guard
# on the heaviest of those before importing the module under test.
pytest.importorskip("lsst.meas.algorithms.maskStreaks", reason="LSST meas.algorithms not available")
afwImage = pytest.importorskip("lsst.afw.image", reason="LSST afw.image not available")

from astro_lfd.meas.detectStreaks import (  # noqa: E402
    KHTDetectConfig,
    KHTDetectTask,
    binary_dilation,
    get_pixel_mask,
)


# --- get_pixel_mask ---------------------------------------------------------
def test_get_pixel_mask_single_plane() -> None:
    """A single plane's bit is reported exactly where it is set."""
    mask = afwImage.Mask(5, 4)
    bit = mask.getPlaneBitMask("DETECTED")
    mask.array[2, 3] |= bit

    out = get_pixel_mask(mask, "DETECTED")
    assert out.dtype == bool
    assert out[2, 3]
    assert out.sum() == 1


def test_get_pixel_mask_multiple_planes_union() -> None:
    """A list of planes yields the union of their set pixels."""
    mask = afwImage.Mask(5, 4)
    mask.array[0, 0] |= mask.getPlaneBitMask("BAD")
    mask.array[1, 1] |= mask.getPlaneBitMask("SAT")

    out = get_pixel_mask(mask, ["BAD", "SAT"])
    assert out[0, 0]
    assert out[1, 1]
    assert out.sum() == 2
    # A plane that was never set contributes nothing.
    assert not get_pixel_mask(mask, "EDGE").any()


# --- binary_dilation --------------------------------------------------------
def test_binary_dilation_radius() -> None:
    """Dilation by radius r turns on all pixels within r of a seed."""
    img = np.zeros((7, 7), dtype=bool)
    img[3, 3] = True

    dilated = binary_dilation(img, 1)
    # The four edge-neighbors (distance 1) switch on; diagonals (sqrt2 > 1) do not.
    assert dilated[3, 3]
    assert dilated[2, 3] and dilated[4, 3] and dilated[3, 2] and dilated[3, 4]
    assert not dilated[2, 2]


def test_binary_dilation_zero_radius_is_identity() -> None:
    """Dilating by radius 0 leaves the seed pixels unchanged."""
    img = np.zeros((4, 4), dtype=bool)
    img[1, 1] = True
    assert np.array_equal(binary_dilation(img, 0), img)


# --- config -----------------------------------------------------------------
def test_config_defaults() -> None:
    """Representative config fields carry their documented defaults."""
    config = KHTDetectConfig()
    assert config.detected_mask_plane == "DETECTED"
    assert config.cluster_minimum_size == 50
    assert config.rho_bin_size == 40.0
    assert config.theta_bin_size == 2.0
    assert "SPIKE" in config.bad_mask_planes


# --- task construction ------------------------------------------------------
def test_task_wires_up() -> None:
    """The task exposes its config class and default name."""
    task = KHTDetectTask()
    assert isinstance(task.config, KHTDetectConfig)
    assert KHTDetectTask.ConfigClass is KHTDetectConfig
    assert KHTDetectTask._DefaultName == "khtDetect"


# --- _cluster_lines ---------------------------------------------------------
def test_cluster_lines_separates_two_groups() -> None:
    """Two well-separated (rho, theta) groups collapse to two centers."""
    task = KHTDetectTask()
    # Two tight clusters far apart in rho (bin size 40) and theta (bin size 2).
    rhos = np.array([100.0, 102.0, 98.0, 900.0, 905.0, 895.0])
    thetas = np.array([10.0, 10.5, 9.5, 80.0, 80.5, 79.5])

    out_rhos, out_thetas = task._cluster_lines(rhos, thetas)

    assert out_rhos.shape == out_thetas.shape
    assert out_rhos.size == 2
    order = np.argsort(out_rhos)
    assert out_rhos[order][0] == pytest.approx(100.0, abs=40.0)
    assert out_rhos[order][1] == pytest.approx(900.0, abs=40.0)
