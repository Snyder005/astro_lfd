import ast
from dataclasses import dataclass, field
from typing import Self

import lsst.afw.image as afwImage
import numpy as np
from numpy.typing import NDArray

from .streak import Streak

NY: float = 4004
NX: float = 4096
FWHM: float = 0.7
READ_NOISE: float = 7.0
PIXEL_SCALE: float = 0.2
SKY_COUNTS: dict[str, float] = {
    "u": 81.0,
    "g": 411.0,
    "r": 819.0,
    "i": 1173.0,
    "z": 1784.0,
    "Y": 2371.0,
}


@dataclass
class SimulatedImage:
    """A simulated image with its variance and mask planes.

    Parameters
    ----------
    image : `numpy.ndarray`, (Ny, Nx)
        Background-subtracted image, in units of ``meta['BUNIT']``.
    variance : `numpy.ndarray`, (Ny, Nx)
        Per-pixel variance, units of ``meta['BUNIT']`` squared.
    mask : `numpy.ndarray`, (Ny, Nx)
        Bit mask, where ``0`` means good.
    meta : `dict`
        Provenance and configuration recorded in the output header.
    """

    image: NDArray[np.float64]
    variance: NDArray[np.float64]
    mask: NDArray[np.int32]
    meta: dict = field(default_factory=dict)

    @classmethod
    def load_npz(cls, infile: str) -> Self:
        d = np.load(infile, allow_pickle=False)
        return cls(d["image"], d["variance"], d["mask"], ast.literal_eval(str(d["meta"])))

    @classmethod
    def simulate_exposure(
        cls,
        streak: Streak,
        band: str = "r",
        shape: tuple[int, int] = (NY, NX),
        read_noise: float = READ_NOISE,
        fwhm: float = FWHM,
        pixel_scale: float = PIXEL_SCALE,
        calib: float = 1.0,
        unit: str = "nJy",
        seed: int | None = None,
    ) -> Self:
        """Simulate a background-subtracted difference image with one streak.

        Parameters
        ----------
        streak : `Streak`
            The streak geometry and brightness.
        band : `str`, optional
            The filter band of the observation (r-band, by default).
        shape : `tuple` [`int`], optional
            The image shape ((4004, 4096), by default)
        read_noise : `float`, optional
            The read noise, in electrons per pixel (7.0, by default).
        fwhm : `float`, optional
            The PSF full-width-at-half-maximum, in arcseconds (0.7, by
            default).
        pixel_scale : `float`, optional
            The pixel scale, in arcseconds per pixel (0.2, by default).
        calib : `float`, optional
            The calibration factor, in electrons per output unit (a constant
            scalar across the frame). For ``unit='ADU'`` this is the amplifier
            gain; for ``unit='nJy'`` it is the photometric calibration factor.
            The default is 1.0.
        unit : `str`, optional
            The output brightness unit label (nJy, by default).
        seed : `int`
            Seed for the random generator (None, by default).

        Returns
        -------
        simulated_image : `astro_lfd.sims.SimulatedImage`
            The simulated image with its variance and mask planes.
        """
        sky = SKY_COUNTS[band]
        signal = streak.get_signal(shape, fwhm / pixel_scale)  # electrons, noise-free
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

        return cls(image, variance, mask, meta)

    def save_npz(self, outfile: str) -> None:
        """Save as an NPZ file.

        Parmaters
        ---------
        outfile : `str`
            Output file name (``.npz`` appended if absent).
        """
        np.savez_compressed(
            path,
            image=self.image,
            variance=self.variance,
            mask=self.mask,
            meta=np.array(repr(self.meta)),
        )

    def to_exposure(self) -> afwImage.ExposureF:
        """Convert to an exposure.

        Returns
        -------
        exposure : `lsst.afw.image.ExposureF`
            Exposure with image, variance, and mask planes plus a header.
        """
        ny, nx = self.image.shape
        exp = afwImage.ExposureF(nx, ny)  # (width, height)
        exp.image.array[:] = self.image
        exp.variance.array[:] = self.variance
        exp.mask.array[:] = self.mask
        md = exp.getMetadata()
        for key, value in self.meta.items():
            if value is not None:
                md.set(key, value)

        return exp
