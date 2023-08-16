import dataclasses
import json
import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
import selector
import skimage
import skimage.io
import torch
import torchvision.utils

import data
import model
import transformation


def load_image(path: str):
    image = skimage.io.imread(path, as_gray=True)
    image = skimage.img_as_ubyte(image)

    return torch.from_numpy(image)


@dataclasses.dataclass
class Element:
    name: str
    acronym: str
    color_hex_triplet: str


def add_elements(dictionary, element):
    dictionary[element['id']] = Element(
        name=element['name'],
        acronym=element['acronym'],
        color_hex_triplet=element['color_hex_triplet'],
    )

    for child in element['children']:
        add_elements(dictionary, child)
    return dictionary


def extract_elements(root):
    return add_elements({}, root)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    dataset = selector.add_arguments(parser, 'dataset', data.CCFv3Torch)()
    parser.add_argument('--annotation_volume', type=str)
    parser.add_argument('--annotation_descriptions', type=str)
    
    parser.add_argument('--model', type=str)
    parser.add_argument('--size', type=int)
    
    parser.add_argument('--slide', type=int)
    parser.add_argument('--dorsor', type=float, default=0)
    parser.add_argument('--medior', type=float, default=0)

    parser.add_argument('--image', type=str)

    args = parser.parse_args()

    annotations = data.CCFv3Torch(header_path=args.annotation_volume, preserve_int=True)
    descriptions = json.load(open(args.annotation_descriptions, 'r'))['msg'][0]

    regions = extract_elements(descriptions)

    net = model.GridRegressor.load_from_checkpoint(args.model, size=args.size)
    net.eval()

    fixed = load_image(args.image).unsqueeze(0)
    moving = dataset._extract_plane(
        slide=args.slide,
        dorsoventral_angle=args.dorsor,
        mediolateral_angle=args.medior
    ).unsqueeze(0)

    fixed = fixed.unsqueeze(1)
    moving = moving.unsqueeze(1)

    fixed = transformation.normalise(fixed)
    moving = transformation.normalise(moving)

    fixed = transformation.pad_resize(fixed, args.size, args.size)
    moving = transformation.pad_resize(moving, args.size, args.size)

    with torch.no_grad():
        x, y, r, s, a = (v.item() for v in net((fixed, moving)))
    print(x, y, r, s, a)

    moved = transformation.affine(moving[0], x, y, r, s, a)
    
    atlas_plate = annotations._extract_plane(
        slide=args.slide,
        dorsoventral_angle=args.dorsor,
        mediolateral_angle=args.medior
    )
    atlas = torch.zeros((3, *atlas_plate.shape))
    for i, row in enumerate(atlas_plate):
        for j, value in enumerate(row):
            if value == 0:
                continue

            element = regions[value.item()]
            hex_code = element.color_hex_triplet
            r = int(hex_code[0:2], 16) / 255
            g = int(hex_code[2:4], 16) / 255
            b = int(hex_code[4:6], 16) / 255

            atlas[0, i, j] = r
            atlas[1, i, j] = g
            atlas[2, i, j] = b


    moved_atlas = transformation.affine(atlas, x, y, r, s, a)
    resized_moved_atlas = transformation.pad_resize(moved_atlas, args.size, args.size)

    torchvision.utils.save_image(
        torch.stack((
            resized_moved_atlas,
            fixed[0].expand(3, -1, -1),
            moved.expand(3, -1, -1),
            moving[0].expand(3, -1, -1),
            transformation.pad_resize(atlas, args.size, args.size),
        )),
        f'reg/{os.path.basename(args.image)}.png'
    )







### Test of using iterative prediction. Not immediately helpful...
    # transformation_matrix = torch.eye(3).float()
    # for i in range(20):
    #     fixed_ = f_t.affine(
    #         fixed[0],
    #         list(transformation_matrix[:2].flatten())
    #     ).unsqueeze(0)

    #     fixed_ = transformation.pad_resize(fixed_, args.size, args.size)
    #     moving = transformation.pad_resize(moving, args.size, args.size)

    #     with torch.no_grad():
    #         x, y, r, s, a = (v.item() for v in net((fixed_, moving)))
    #     print(x, y, r, s, a)

    #     transformation_matrix = transformation._translation_matrix(x, y) @ transformation._rotation_matrix(r) @ transformation._scale_matrix(s, a) @ transformation_matrix

    #     # moved = transformation.affine(moving[0], x, y, r, s, a)
    #     moved = f_t.affine(
    #         moving[0],
    #         list(torch.linalg.inv(transformation_matrix)[:2].flatten())
    #     )

    #     torchvision.utils.save_image(
    #         torch.stack((
    #             # transformation.pad_resize(fixed, args.size, args.size).squeeze(),
    #             fixed_.squeeze(),
    #             moved.squeeze(),
    #             moving.squeeze(),
    #         )).unsqueeze(1),
    #         f'reg/{os.path.basename(args.image)}_{i}.png'
    #     )
