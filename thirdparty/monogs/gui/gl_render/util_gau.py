from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray

    def flat(self) -> np.ndarray:
        ret = np.concatenate(
            [self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1
        )
        return np.ascontiguousarray(ret)

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        return self.sh.shape[-1]
