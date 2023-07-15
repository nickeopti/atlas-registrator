import random
from typing import Optional, Tuple

import nrrdu
import numpy as np
import scipy.spatial
import torch.utils.data
import torchvision.transforms

import transformation


class CCFv3(torch.utils.data.Dataset):
    def __init__(
        self,
        header_path: str,
        binary_path: Optional[str] = None,
        size: Tuple[int, int] = (256, 256),
        exclude_ends: Tuple[int, int] = (0, 0),
        dorsoventral_rotation: Tuple[float, float] = (-25, 25),
        mediolateral_rotation: Tuple[float, float] = (-8, 8),
        augment: bool = True,
    ) -> None:
        self.volume = nrrdu.read(header_path, binary_path)
        self.volume /= self.volume.max()

        self.pad = torchvision.transforms.Pad(150)
        self.size = size
        self.exclude_ends = exclude_ends
        self.dorsoventral_rotation = dorsoventral_rotation
        self.medialateral_rotation = mediolateral_rotation
        self.augment = augment

        dim = self.volume.shape
        xx, yy = map(np.arange, dim[1:])
        self.g = np.stack(
            np.meshgrid(xx - dim[1] // 2, yy - dim[2] // 2),
            -1
        ).reshape(-1, 2)

    def _extract_plane(self, slide, dorsoventral_angle, mediolateral_angle):
        r = scipy.spatial.transform.Rotation.from_euler(
            'zy', [dorsoventral_angle, mediolateral_angle], degrees=True
        )
        R = r.as_matrix()

        dim = self.volume.shape

        z = np.zeros((dim[1] * dim[2], 1))
        g = np.hstack([z, self.g])

        p = (R @ g.T).T.round().astype(int)

        values = self.volume[(
            np.clip(p[:, 0] + slide, 0, dim[0]-1),
            np.clip(p[:, 1] + dim[1] // 2, 0, dim[1]-1),
            np.clip(p[:, 2] + dim[2] // 2, 0, dim[2]-1)
        )].reshape(dim[1], dim[2], order='F')

        slide = values

        return slide

    def __len__(self) -> int:
        return self.volume.shape[0] - sum(self.exclude_ends)

    def __getitem__(self, index):
        z = index + self.exclude_ends[0]
        d = random.uniform(*self.dorsoventral_rotation)
        m = random.uniform(*self.medialateral_rotation)

        r = random.uniform(-30, 30)
        s = random.uniform(0.5, 2)
        a = random.uniform(0.5, 2)

        x = random.randint(-149, 149)
        y = random.randint(-149, 149)

        fixed = self._extract_plane(
            z + random.randint(-2, 2),
            d + random.uniform(-1, 1),
            m + random.uniform(-1, 1)
        )
        fixed = torch.from_numpy(fixed)

        moving = self._extract_plane(z, d, m)
        moving = torch.from_numpy(moving)

        fixed = self.pad(fixed.unsqueeze(0))
        moving = self.pad(moving.unsqueeze(0))

        fixed = transformation.affine(
            fixed,
            angle=r,
            translate=(x, y),
            scale=(s, s * a),
            shear=(0, 0)
        )
        fixed = transformation.pad_resize(fixed, *self.size[::-1]).squeeze()
        moving = transformation.pad_resize(moving, *self.size[::-1]).squeeze()

        return fixed.float(), moving.float(), x, y, r, s, a
