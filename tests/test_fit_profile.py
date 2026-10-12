"""Tests for the `FitProfile` line profile measurement plugin.

A synthetic Moffat streak with white noise is fit from a perturbed initial
line. The fit must recover the truth in the exposure PIXEL frame, including
for exposures with a non-zero origin (``xy0``).
"""

import numpy as np
import pytest

geom = pytest.importorskip("lsst.geom", reason="requires the LSST stack")
afwImage = pytest.importorskip("lsst.afw.image", reason="requires the LSST stack")

from astro_lfd.algorithms.plugins import FitProfile  # noqa: E402
from astro_lfd.geom import Line2D  # noqa: E402

NX, NY = 600, 500
SIGMA = 4.0
PEAK = 20.0


def _make_exposure(line, xy0=geom.Point2I(0, 0), seed=42):
    """Return an exposure with a Moffat streak along ``line`` (PIXEL frame)."""
    rng = np.random.default_rng(seed)
    x = xy0.x + np.arange(NX)
    y = xy0.y + np.arange(NY)
    distance = line.normal_coordinate_array(x[None, :], y[:, None])
    image = PEAK * (1 + (distance / SIGMA) ** 2) ** -2.5 + rng.normal(0.0, 1.0, (NY, NX))

    exposure = afwImage.ExposureF(geom.Box2I(xy0, geom.Extent2I(NX, NY)))
    exposure.image.array[:] = image
    exposure.variance.array[:] = 1.0
    exposure.mask.array[:] = 0
    for plane in ("ITL_DIP", "SPIKE"):
        exposure.mask.addMaskPlane(plane)
    detected = exposure.mask.getPlaneBitMask("DETECTED")
    exposure.mask.array[np.abs(distance) < 2 * SIGMA] |= detected
    return exposure


def _truth(xy0=geom.Point2I(0, 0)):
    """Truth line through the image center, in the PIXEL frame."""
    center = geom.Point2D(xy0.x + NX / 2, xy0.y + NY / 2)
    direction = geom.Extent2D(np.cos(np.deg2rad(25.0)), np.sin(np.deg2rad(25.0)))
    return Line2D.from_point_and_direction(center, direction)


def _perturbed(line):
    return Line2D(line.rho + 2.0, line.theta + 0.3 * geom.degrees)


def test_recovers_truth():
    truth = _truth()
    result = FitProfile().measure(_make_exposure(truth), _perturbed(truth))
    assert result is not None
    assert abs(result.line.rho - truth.rho) < 0.1
    assert abs((result.line.theta - truth.theta).asDegrees()) < 0.01
    assert abs(result.sigma - SIGMA) < 0.1
    assert result.mask.shape == (NY, NX)
    assert result.mask.any()


def test_consistent_across_xy0():
    # The Newton steps depend on the PIXEL origin through the rho-theta
    # coupling, so results agree to well below the fit precision, not exactly.
    results = []
    for xy0 in (geom.Point2I(0, 0), geom.Point2I(1500, -700)):
        offset = geom.Extent2D(geom.Point2D(xy0))
        truth = _truth(xy0)
        result = FitProfile().measure(_make_exposure(truth, xy0), _perturbed(_truth()).translated(offset))
        assert result is not None
        results.append((result, offset))

    (r0, _), (r1, offset) = results
    line1 = r1.line.translated(-offset)
    assert line1.rho == pytest.approx(r0.line.rho, abs=1e-2)
    assert line1.theta.asDegrees() == pytest.approx(r0.line.theta.asDegrees(), abs=1e-3)
    assert r1.sigma == pytest.approx(r0.sigma, rel=1e-3)
    assert r1.reduced_chi2 == pytest.approx(r0.reduced_chi2, rel=1e-3)
    np.testing.assert_array_equal(r1.mask, r0.mask)


def test_line_outside_image_fails():
    truth = _truth()
    outside = Line2D(truth.rho + 5000.0, truth.theta)
    assert FitProfile().measure(_make_exposure(truth), outside) is None
