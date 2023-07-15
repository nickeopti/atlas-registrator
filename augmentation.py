"""
All augmentations work on batches, but expect explicit channel dimension,
i.e., tensors shall have dimensions [B, H, W], and hence work for
grey-scale images only.
"""

import random
from typing import Callable, Iterable

import kornia.filters
import numpy as np
import torch


def additive_gaussian(x: torch.Tensor, mean: float = 0, sd: float = 0.01):
    noise = torch.normal(mean=mean, std=sd, size=x.shape, device=x.device)

    return x + noise


def multiplicative_gaussian(x: torch.Tensor, mean: float = 0, sd: float = 0.01):
    noise = torch.normal(mean=mean, std=sd, size=x.shape, device=x.device)

    return x * (1 + noise)


def gaussian_blur(x: torch.Tensor):
    kernel_size = random.randint(1, 3) * 2 + 1
    sigma = random.uniform(0.1, 2)

    return kornia.filters.gaussian_blur2d(x.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma))[:, 0, :, :]


def batch_checkerboard(x: torch.Tensor, size_range: tuple[int, int] = (50, 125)):
    yy, xx = torch.from_numpy(np.indices(x.shape[-2:])).to(device=x.device)

    b, h, w = x.shape
    sizes = torch.randint(*size_range, (b, ), device=x.device)

    xx -= w // 2
    yy -= h // 2

    alpha = torch.rand(b, device=x.device) - 0.5
    xx, yy = xx + alpha[:, None, None] * yy, \
             yy - alpha[:, None, None] * xx

    xx = xx // sizes[:, None, None]
    yy = yy // sizes[:, None, None]

    scale = 1 - (torch.abs(xx) / xx.max(dim=-1).values.max(dim=-1).values[:, None, None])**2 - yy % 2 / 5

    return x * scale


def batch_black(x: torch.Tensor):
    yy, xx = torch.from_numpy(np.indices(x.shape[-2:])).to(device=x.device)

    b, h, w = x.shape

    alpha = torch.rand(b, device=x.device) - 0.5
    beta = torch.randint(1, h // 3, (b,), device=x.device)
    top = yy + alpha[:, None, None] * xx < beta[:, None, None]

    alpha = torch.rand(b, device=x.device) - 0.5
    beta = torch.randint(1, h // 3, (b,), device=x.device)
    bottom = yy + alpha[:, None, None] * xx > h - beta[:, None, None]

    alpha = torch.rand(b, device=x.device) - 0.5
    beta = torch.randint(1, w // 3, (b,), device=x.device)
    left = xx + alpha[:, None, None] * yy < beta[:, None, None]

    alpha = torch.rand(b, device=x.device) - 0.5
    beta = torch.randint(1, h // 3, (b,), device=x.device)
    right = xx + alpha[:, None, None] * yy > w - beta[:, None, None]

    return x * (~top) * (~bottom) * (~left) * (~right)


def batch_erase(x: torch.Tensor, n: int = 5, size_range: tuple[int, int] = (50, 125)):
    b = (x.shape[0], )
    ns = torch.randint(1, n, b, device=x.device)
    sizes = torch.randint(*size_range, b, device=x.device)

    for i, (n, size) in enumerate(zip(ns, sizes)):
        rs = torch.randint(0, x.shape[1] - size, (n,), device=x.device)
        cs = torch.randint(0, x.shape[2] - size, (n,), device=x.device)

        for r, c in zip(rs, cs):
            x[i, r:r + size, c:c + size] = 0
    
    return x


def batch_horizontal_gradient(x: torch.Tensor):
    _, xx = torch.from_numpy(np.indices(x.shape[-2:])).to(device=x.device)
    xx = xx.float()
    m = torch.max(xx)

    flip = torch.bernoulli(torch.full((x.shape[0],), 0.5)).to(device=x.device)
    xx = torch.abs(m * flip[:, None, None] - xx)

    s = torch.distributions.Uniform(2, 5).sample((x.shape[0], )).to(device=x.device)
    shade = xx / (m * s[:, None, None])

    return x * (1 - shade)


def batch_vertical_gradient(x: torch.Tensor):
    yy, _ = torch.from_numpy(np.indices(x.shape[-2:])).to(device=x.device)
    yy = yy.float()
    m = torch.max(yy)

    flip = torch.bernoulli(torch.full((x.shape[0],), 0.5)).to(device=x.device)
    yy = torch.abs(m * flip[:, None, None] - yy)

    s = torch.distributions.Uniform(2, 5).sample((x.shape[0], )).to(device=x.device)
    shade = yy / (m * s[:, None, None])

    return x * (1 - shade)


def compose(universe, probability_all_sampled: float = 0.5):
    p = probability_all_sampled**(1 / len(universe))

    sampled = [aug for aug in universe if random.random() < p]
    random.shuffle(sampled)  # random inplace permutation

    return sampled


def augment(x: torch.Tensor, augmentations: Iterable[Callable[[torch.Tensor], torch.Tensor]]):
    for augmentation in augmentations:
        x = augmentation(x)

    return x


ALL_AUGMENTATIONS = (
    additive_gaussian,
    multiplicative_gaussian,
    gaussian_blur,
    batch_checkerboard,
    batch_black,
    batch_erase,
    batch_vertical_gradient,
    batch_horizontal_gradient,
)


if __name__ == '__main__':
    import sys

    import matplotlib.pyplot as plt
    
    slide = torch.from_numpy(np.load(sys.argv[1]))
    slide /= slide.max()
    print(slide.shape)

    def view(image):
        plt.imshow(image, cmap='gray')
        plt.show()


    view(slide)

    while True:
        augmented = augment(
            torch.clone(slide.expand(5, *slide.shape)),
            compose(ALL_AUGMENTATIONS, 0.2)
        )
        for image in augmented:
            view(image.numpy())
