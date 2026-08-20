import ast
from dataclasses import asdict, dataclass, field
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
    def from_exposure(cls, exposure: afwImage.ExposureF) -> Self:
        """Create a `SimulatedImage` instance from an exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            The streak exposure.

        Returns
        -------
        simulated_image : `astro_lfd.sims.SimulatedImage`
            The simulated image with its variance and mask planes.
        """
        md = exposure.getMetadata()
        return cls(
            exposure.image.array.copy(),
            exposure.variance.array.copy(),
            exposure.mask.array.copy(),
            {key: md.get(key) for key in md.names()},
        )

    @classmethod
    def load_npz(cls, infile: str) -> Self:
        """Create a `SimulatedImage` instance by loading an NPZ file.

        Parameters
        ----------
        infile : `str`
            The input file name.

        Returns
        -------
        simulated_image : `astro_lfd.sims.SimulatedImage`
            The simulated image with its variance and mask planes.
        """
        d = np.load(infile, allow_pickle=False)
        return cls(d["image"], d["variance"], d["mask"], ast.literal_eval(str(d["meta"])))

    @classmethod
    def simulate(
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
        """Create a `SimulatedImage` instance by simulating an image with a
        single streak.

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
            The seed for the random generator (None, by default).

        Returns
        -------
        simulated_image : `astro_lfd.sims.SimulatedImage`
            The simulated image with its variance and mask planes.
        """
        sky = SKY_COUNTS[band]
        signal = streak.get_signal(shape, fwhm=fwhm / pixel_scale)
        rng = np.random.default_rng(seed)

        image = rng.normal(0.0, read_noise, size=shape)
        image += rng.poisson(sky, size=shape) - sky
        image += rng.poisson(signal)
        variance = np.full(shape, read_noise**2) + sky + signal

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

    def to_exposure(self) -> afwImage.ExposureF:
        """Convert to an exposure.

        Returns
        -------
        exposure : `lsst.afw.image.ExposureF`
            An exposure with image, variance, and mask planes plus a header.
        """
        ny, nx = self.image.shape
        exposure = afwImage.ExposureF(nx, ny)
        exposure.image.array[:] = self.image
        exposure.variance.array[:] = self.variance
        exposure.mask.array[:] = self.mask

        md = exposure.getMetadata()
        for key, value in self.meta.items():
            if value is not None:
                md.set(key, value)

        return exposure

    def write_fits(self, outfile: str) -> None:
        """Write as an LSST-compatible multi-plane FITS file.

        Parameters
        ----------
        outfile : `str`
            The output file name.
        """
        self.to_exposure().writeFits(outfile)

    def write_npz(self, outfile: str) -> None:
        """Write as an NPZ file.

        Parmaters
        ---------
        outfile : `str`
            The output file name (``.npz`` appended if absent).
        """
        np.savez_compressed(
            path,
            image=self.image,
            variance=self.variance,
            mask=self.mask,
            meta=np.array(repr(self.meta)),
        )
