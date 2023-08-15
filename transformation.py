import kornia.enhance.equalization
import torch
import torchvision.transforms._functional_tensor as f_t
import torchvision.transforms.functional as f


def clahe(x: torch.Tensor):
    return kornia.enhance.equalization.equalize_clahe(
        x / x.max(dim=-1).values.max(dim=-1).values[:, :, None, None],
        clip_limit=5.0
    )


def normalise(x: torch.Tensor):
    return x / x.max(dim=-1).values.max(dim=-1).values[:, :, None, None]


def pad_resize(image: torch.Tensor, h, w):
    *_, h_1, w_1 = image.shape
    ratio_f = w / h
    ratio_1 = w_1 / h_1

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):
        # padding to preserve aspect ratio
        hp = int(w_1 / ratio_f - h_1)
        wp = int(ratio_f * h_1 - w_1)
        if hp > 0 and wp < 0:
            hp = hp // 2
            image = f.pad(image, (0, hp, 0, hp), 0, "constant")
        elif hp < 0 and wp > 0:
            wp = wp // 2
            image = f.pad(image, (wp, 0, wp, 0), 0, "constant")

    return f.resize(image, [h, w], antialias=True)


def _differentiable_affine_matrix(rows):
    assert len(rows) == 3

    matrix = torch.eye(3).double()  # identity matrix as basis
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            # in-place, gradient preserving update
            matrix.index_put_(
                (torch.tensor(i), torch.tensor(j)),
                values=value.double() if isinstance(value, torch.Tensor) else torch.tensor(value).double()
            )

    return matrix


def _vertical_reflection_matrix():
    return torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]).float()


def _translation_matrix(x: float, y: float):
    return _differentiable_affine_matrix([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ]).float()
    # m = torch.eye(3).float()
    # m[0, 2] = x
    # m[1, 2] = y
    # return m


def _rotation_matrix(angle: float):
    if isinstance(angle, (int, float)):
        angle = torch.tensor(angle).float()
    theta = torch.deg2rad(angle)
    return _differentiable_affine_matrix([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ]).float()
    # m = torch.eye(3).float()
    # m[0, 0] = torch.cos(theta)
    # m[0, 1] = -torch.sin(theta)
    # m[1, 0] = torch.sin(theta)
    # m[1, 1] = torch.cos(theta)
    # return m


def _scale_matrix(s: float, a: float):
    return _differentiable_affine_matrix([
        [s, 0, 0],
        [0, s * a, 0],
        [0, 0, 1]
    ]).float()
    # m = torch.eye(3).float()
    # m[0, 0] = s
    # m[1, 1] = s * a
    # return m


def affine_transformation_matrix(x: float, y: float, angle: float, scale: float, aspect: float, width: int, height: int):
    return (
        # go from cartesian to image coordinates
        _translation_matrix(0, height) @
        _vertical_reflection_matrix() @
        # undo centering
        _translation_matrix(width / 2, height / 2) @
        # translate
        _translation_matrix(x, -y) @
        # rotate
        _rotation_matrix(-angle) @
        # scale
        _scale_matrix(scale, aspect) @
        # centering
        _translation_matrix(-width / 2, -height / 2) @
        # go from image to cartesian coordinates
        _vertical_reflection_matrix() @
        _translation_matrix(0, -height)
    )


def affine_transformation_matrix_torch(x: float, y: float, angle: float, scale: float, aspect: float):
    t = _translation_matrix(x, y)
    r = _rotation_matrix(angle)
    s = _scale_matrix(scale, aspect)

    matrix = t @ r @ s
    inverse = torch.linalg.inv(matrix)

    return list(inverse[:2].flatten())


def affine(img: torch.Tensor, x: float, y: float, angle: float, scale: float, aspect: float):
    matrix = affine_transformation_matrix_torch(x, y, angle, scale, aspect)

    return f_t.affine(img, matrix=matrix)
