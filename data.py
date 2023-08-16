import random
from typing import Optional, Tuple

import nrrdu
import scipy.spatial
import torch.utils.data

import transformation


class CCFv3Torch(torch.utils.data.Dataset):
    def __init__(
        self,
        header_path: str,
        binary_path: Optional[str] = None,
        exclude_ends: Tuple[int, int] = (0, 0),
        dorsoventral_rotation: Tuple[float, float] = (-25, 25),
        mediolateral_rotation: Tuple[float, float] = (-8, 8),
        augment: bool = True,
        accelerator: str = 'cpu',
        preserve_int: bool = False,
    ) -> None:
        self.device = accelerator

        self.volume = nrrdu.read(header_path, binary_path)
        if preserve_int:
            self.volume = self.volume.astype(int)
        self.volume = torch.from_numpy(self.volume).to(self.device)
        if not preserve_int:
            self.volume /= self.volume.max()

        self.exclude_ends = exclude_ends
        self.dorsoventral_rotation = dorsoventral_rotation
        self.medialateral_rotation = mediolateral_rotation
        self.augment = augment

        dim = self.volume.shape
        xx, yy = map(torch.arange, dim[1:])
        self.g = torch.stack(
            torch.meshgrid(xx - dim[1] // 2, yy - dim[2] // 2, indexing='ij'),
            -1
        ).reshape(-1, 2).to(self.device)

    def _extract_plane(self, slide, dorsoventral_angle, mediolateral_angle):
        r = scipy.spatial.transform.Rotation.from_euler(
            'zy', [dorsoventral_angle, mediolateral_angle], degrees=True
        )
        R = torch.from_numpy(r.as_matrix()).to(self.device).float()

        dim = self.volume.shape

        z = torch.zeros((dim[1] * dim[2], 1), device=self.device)
        g = torch.hstack([z, self.g])

        p = (R @ g.T).T.round().type(torch.int)

        values = self.volume[(
            torch.clip(p[:, 0] + slide, 0, dim[0] - 1),
            torch.clip(p[:, 1] + dim[1] // 2, 0, dim[1] - 1),
            torch.clip(p[:, 2] + dim[2] // 2, 0, dim[2] - 1)
        )].reshape(dim[1], dim[2])

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
        a = random.uniform(2 / 3, 1.5)

        x = random.randint(-149, 149)
        y = random.randint(-149, 149)

        fixed = self._extract_plane(
            z + random.randint(-3, 3),
            d + random.uniform(-1, 1),
            m + random.uniform(-1, 1)
        )

        moving = self._extract_plane(z, d, m)

        fixed = fixed.unsqueeze(0)
        moving = moving.unsqueeze(0)

        fixed = transformation.affine(
            fixed,
            x=x,
            y=y,
            angle=r,
            scale=s,
            aspect=a
        )

        return fixed.float().squeeze(), moving.float().squeeze(), x, y, r, s, a
