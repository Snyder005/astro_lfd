"""Synthetic test-image generation for LFD (Linear Feature Detector) development.

Simulates an LSST-Camera-like difference image containing a single straight
streak, plus its variance and mask planes.

The radiometric simulation is always done in **electrons** because the noise
model requires it: sky and streak shot noise are Poisson (valid only on
electron counts) and read noise is quoted in electrons.  The final image and
variance are then converted to the requested output ``unit`` via a single
scalar calibration factor ``calib`` (electrons per output unit).  This scale
does not affect ADRT detection, which operates on the significance image
``D / sqrt(V)`` (invariant under the conversion); it only controls output
realism and compatibility.

I/O is provided in two formats: compact ``.npz`` (no LSST dependency) for numpy
and ADRT development, and ``lsst.afw.image.ExposureF`` FITS (LSST Science
Pipelines compatible).  Converters between the two are included.  The afw
adapter is behind a lazy import so this module has no hard LSST dependency.
"""

from __future__ import annotations

import ast
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import erf

# --- Default instrument constants (LSST Camera science sensor) --------------
NX, NY = 4096, 4004  # pixels (columns, rows)
READ_NOISE = 7.0  # electrons per pixel
FWHM = 0.7  # arcseconds
PIXEL_SCALE = 0.2  # arcseconds per pixel
SKY_COUNTS = {  # electrons per pixel, per band
    "u": 81.0,
    "g": 411.0,
    "r": 819.0,
    "i": 1173.0,
    "z": 1784.0,
    "Y": 2371.0,
}
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclass
class StreakConfig:
    """Streak geometry and brightness in Hesse normal form.

    The line satisfies ``rho = x*cos(theta) + y*sin(theta)`` in pixel
    coordinates; the cross-section profile is a top-hat convolved with the PSF.

    Parameters
    ----------
    theta : float
        Line orientation in degrees.
    rho : float
        Perpendicular offset of the line from the origin, in pixels.
    width : float
        Top-hat full width of the streak, in pixels.
    peak_signal : float
        Signal at the ridge center, in electrons.
    """

    theta: float = 0.0
    rho: float = 2000.0
    width: float = 20.0
    peak_signal: float = 1000.0


@dataclass
class TestImage:
    """A synthetic difference image with its variance and mask planes.

    Attributes
    ----------
    image : numpy.ndarray
        Background-subtracted difference image, ``float32``, in ``meta['BUNIT']``.
    variance : numpy.ndarray
        Per-pixel variance, ``float32``, in ``BUNIT`` squared.
    mask : numpy.ndarray
        Bit mask, ``int32``; ``0`` means good.
    meta : dict
        Provenance and configuration recorded in the output header.
    """

    image: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    meta: dict = field(default_factory=dict)


def blurred_box(x: np.ndarray, width: float, sigma: float) -> np.ndarray:
    """Top-hat convolved with a Gaussian, peak-normalized to 1.

    Parameters
    ----------
    x : numpy.ndarray
        Signed perpendicular distance from the line center, in pixels.
    width : float
        Top-hat full width, in pixels.
    sigma : float
        Gaussian standard deviation (PSF), in pixels.

    Returns
    -------
    numpy.ndarray
        Profile values in ``[0, 1]``, equal to 1 at the ridge center.
    """
    y = 0.5 * (erf((x + width / 2) / (np.sqrt(2) * sigma)) - erf((x - width / 2) / (np.sqrt(2) * sigma)))
    return y / erf(width / (2 * np.sqrt(2) * sigma))


def streak_signal(shape: tuple[int, int], streak: StreakConfig, sigma: float) -> np.ndarray:
    """Noise-free streak signal (electrons) for each pixel from its line distance.

    Parameters
    ----------
    shape : tuple of int
        Image shape ``(ny, nx)``.
    streak : StreakConfig
        Streak geometry and brightness.
    sigma : float
        PSF standard deviation, in pixels.

    Returns
    -------
    numpy.ndarray
        Noise-free signal, in electrons, with shape ``shape``.
    """
    ny, nx = shape
    theta = np.deg2rad(streak.theta)
    gy, gx = np.ogrid[:ny, :nx]
    distance = gx * np.cos(theta) + gy * np.sin(theta) - streak.rho
    return blurred_box(distance, streak.width, sigma) * streak.peak_signal


def simulate_exposure(
    streak: StreakConfig,
    band: str = "i",
    shape: tuple[int, int] = (NY, NX),
    read_noise: float = READ_NOISE,
    fwhm: float = FWHM,
    pixel_scale: float = PIXEL_SCALE,
    calib: float = 1.0,
    unit: str = "nJy",
    seed: int | None = None,
) -> TestImage:
    """Simulate a background-subtracted difference image with one streak.

    The radiometric simulation runs in electrons: read noise (Gaussian), sky
    (Poisson, then mean-subtracted so the background is ~0), and streak shot
    noise (Poisson).  The image and variance are then converted to ``unit`` by
    dividing by the scalar calibration factor ``calib`` (electrons per output
    unit); the variance is divided by ``calib**2``.  Using ``calib=1`` with
    ``unit='electron'`` reproduces the original notebook exactly.

    Parameters
    ----------
    streak : StreakConfig
        Streak geometry and brightness (electrons).
    band : str
        Filter band; selects the sky level from ``SKY_COUNTS``.
    shape : tuple of int
        Image shape ``(ny, nx)``.
    read_noise : float
        Read noise, in electrons per pixel.
    fwhm : float
        PSF FWHM, in arcseconds.
    pixel_scale : float
        Pixel scale, in arcseconds per pixel.
    calib : float
        Calibration factor, in electrons per output unit (a constant scalar
        across the frame). For ``unit='ADU'`` this is the amplifier gain; for
        ``unit='nJy'`` it is the photometric calibration factor.
    unit : str
        Output brightness unit label, recorded as ``BUNIT``.
    seed : int or None
        Seed for the random generator; ``None`` draws a fresh seed.

    Returns
    -------
    TestImage
        Image and variance as ``float32`` in ``unit``, an empty ``int32`` mask,
        and a ``meta`` dict of provenance.
    """
    sigma = (fwhm / pixel_scale) * FWHM_TO_SIGMA
    sky = SKY_COUNTS[band]
    signal = streak_signal(shape, streak, sigma)  # electrons, noise-free
    rng = np.random.default_rng(seed)

    # Image plane (electrons): read noise + sky (mean-subtracted) + streak.
    image = rng.normal(0.0, read_noise, size=shape)
    image += rng.poisson(sky, size=shape) - sky
    image += rng.poisson(signal)

    # Variance plane (electrons^2): read noise + sky + streak (Poisson terms).
    variance = np.full(shape, read_noise**2) + sky + signal

    # Convert electrons -> output unit (variance scales as the square).
    image = (image / calib).astype(np.float32)
    variance = (variance / calib**2).astype(np.float32)
    mask = np.zeros(shape, dtype=np.int32)

    meta = {
        "BUNIT": unit,
        "CALIB": calib,
        "BAND": band,
        "SEED": seed,
        "RDNOISE": read_noise,
        "FWHM": fwhm,
        "PIXSCALE": pixel_scale,
        **{f"STRK_{k.upper()}": v for k, v in asdict(streak).items()},
    }
    return TestImage(image, variance, mask, meta)


# --- I/O: compact npz (no LSST dependency) ----------------------------------
def save_npz(ti: TestImage, path: str) -> None:
    """Save a :class:`TestImage` with :func:`numpy.savez_compressed`.

    Parameters
    ----------
    ti : TestImage
        Image to serialize.
    path : str
        Output path (``.npz`` appended by numpy if absent).
    """
    np.savez_compressed(
        path,
        image=ti.image,
        variance=ti.variance,
        mask=ti.mask,
        meta=np.array(repr(ti.meta)),
    )


def load_npz(path: str) -> TestImage:
    """Load a :class:`TestImage` written by :func:`save_npz`.

    Parameters
    ----------
    path : str
        Path to a ``.npz`` file.

    Returns
    -------
    TestImage
        The deserialized image.
    """
    d = np.load(path, allow_pickle=False)
    return TestImage(d["image"], d["variance"], d["mask"], ast.literal_eval(str(d["meta"])))


# --- Optional afw adapter (isolated LSST dependency, lazy import) -----------
def to_exposure(ti: TestImage):
    """Build an ``lsst.afw.image.ExposureF`` from a :class:`TestImage`.

    Requires ``lsst.afw.image``, imported lazily so the core module has no hard
    LSST dependency (see ``CLAUDE.md``).

    Parameters
    ----------
    ti : TestImage
        Image to convert.

    Returns
    -------
    lsst.afw.image.ExposureF
        Exposure with image, variance, and mask planes plus a header.
    """
    import lsst.afw.image as afwImage

    ny, nx = ti.image.shape
    exp = afwImage.ExposureF(nx, ny)  # (width, height)
    exp.image.array[:] = ti.image
    exp.variance.array[:] = ti.variance
    exp.mask.array[:] = ti.mask
    md = exp.getMetadata()
    for key, value in ti.meta.items():
        if value is not None:
            md.set(key, value)
    return exp


def save_fits(ti: TestImage, path: str) -> None:
    """Write an LSST-compatible multi-plane FITS via ``ExposureF.writeFits``.

    Parameters
    ----------
    ti : TestImage
        Image to write.
    path : str
        Output FITS path.
    """
    to_exposure(ti).writeFits(path)


def from_exposure(exp) -> TestImage:
    """Convert an ``lsst.afw.image.Exposure`` back to a :class:`TestImage`.

    Parameters
    ----------
    exp : lsst.afw.image.Exposure
        Source exposure.

    Returns
    -------
    TestImage
        Copies of the image, variance, and mask planes plus header metadata.
    """
    md = exp.getMetadata()
    return TestImage(
        exp.image.array.copy(),
        exp.variance.array.copy(),
        exp.mask.array.copy(),
        {key: md.get(key) for key in md.names()},
    )
