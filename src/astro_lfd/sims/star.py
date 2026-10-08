from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Star:

    x: float
    y: float
    peak_signal: float
    sigma: float

    def get_signal(self, shape: tuple[int, int]) -> NDArray[np.float32]:
        ny, nx = shape
        gy, gx = np.ogrid[:ny, :nx]
        profile = np.exp(-((gx - self.x) ** 2 + (gy - self.y) ** 2) / (2 * self.sigma**2))
        return profile * self.peak_signal
