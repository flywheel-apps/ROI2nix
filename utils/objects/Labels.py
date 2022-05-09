from dataclasses import dataclass
import numpy as np


@dataclass
class RoiLabel:
    label: str
    index: int
    color: str
    RGB: list
    num_voxels: int = 0
    volume: float = 0.0

    def calc_volume(self, affine):
        self.volume = self.num_voxels * np.abs(np.linalg.det(affine[:3, :3]))
