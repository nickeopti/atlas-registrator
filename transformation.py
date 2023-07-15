import math
from typing import List

import kornia.enhance.equalization
import torch
import torchvision.transforms._functional_tensor as f_t
import torchvision.transforms.functional as f


def clahe(x: torch.Tensor):
    return kornia.enhance.equalization.equalize_clahe(
        x / x.max(dim=-1).values.max(dim=-1).values[:, :, None, None],
        clip_limit=5.0
    )


def pad_resize(image: torch.Tensor, h, w):
    _, h_1, w_1 = image.shape
    ratio_f = w / h
    ratio_1 = w_1 / h_1

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):
        # padding to preserve aspect ratio
        hp = int(w_1/ratio_f - h_1)
        wp = int(ratio_f * h_1 - w_1)
        if hp > 0 and wp < 0:
            hp = hp // 2
            image = f.pad(image, (0, hp, 0, hp), 0, "constant")
        elif hp < 0 and wp > 0:
            wp = wp // 2
            image = f.pad(image, (wp, 0, wp, 0), 0, "constant")

    return f.resize(image, [h, w], antialias=True)


def _get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: List[float], shear: List[float]
) -> List[float]:
    # Adapted from:
    # https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional.py#L1014-L1071
    # to allow for different scales in each direction.

    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    scale_x, scale_y = scale

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear 
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [
        d / scale_x, -b / scale_x, 0.0,
        -c / scale_y, a / scale_y, 0.0
    ]
    # matrix = [x / scale for x in matrix]
    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def affine(img, angle: float, translate: List[int], scale: List[float], shear: List[float]):
    center_f = [0.0, 0.0]
    translate_f = [1.0 * t for t in translate]

    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    return f_t.affine(img, matrix=matrix)
