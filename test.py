from argparse import ArgumentParser

import numpy as np
import skimage
import skimage.exposure
import skimage.io
import torch
import torchvision.transforms.functional as f

import data
import transformation
from model import CascadeRegressor, FullRegressor


def load_image(path: str, scale: float, h, w):
    image = skimage.io.imread(path, as_gray=True)
    image = skimage.img_as_ubyte(image).astype(np.int16)
    image = torch.from_numpy(image)
    # image = image.expand(3, *image.shape)
    image = f.pad(image, 150)

    transformed_image = f.affine(
        image.unsqueeze(0),
        angle=0,
        translate=(0, 0),
        scale=scale,
        shear=(0, 0),
    )

    padded = transformation.pad_resize(transformed_image, w, h)
    
    return padded.expand(3, h, w).numpy()[0]

    # return f.resize(padded, (512, 512)).numpy()[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--cascade', action='store_true')
    parser.add_argument('--cascade_layers', type=int, default=1)
    parser.add_argument('--image', type=str)
    parser.add_argument('--atlas', type=str)
    parser.add_argument('--slide', type=int)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--new', action='store_true')
    args = parser.parse_args()

    size = (256, 256) if args.new else (512, 512)
    atlas = data.CCFv3(args.atlas, dorsoventral_rotation=(-0.0, 0.0), mediolateral_rotation=(-0.0, 0.0), size=size)

    if args.cascade:
        model = CascadeRegressor.load_from_checkpoint(args.model, n=args.cascade_layers, map_location=torch.device('cpu'))
    else:
        model = FullRegressor.load_from_checkpoint(args.model, map_location=torch.device('cpu'))

    image = load_image(args.image, args.scale, *size[::-1])
    print(image.shape)

    with torch.no_grad():
        if args.new:
            fixed = image
            moving = atlas[args.slide][1]
        else:
            fixed = torch.from_numpy(
                skimage.exposure.equalize_adapthist(
                    atlas[args.slide][1].numpy(), nbins=128, clip_limit=0.05
                )
            )
            moving = torch.torch.from_numpy(
                skimage.exposure.equalize_adapthist(
                    image, nbins=128, clip_limit=0.05
                )
            )
            print(fixed.shape, moving.shape)

        prediction = model((fixed.expand(3, *size).unsqueeze(0).float(), moving.expand(3, *size).unsqueeze(0).float()))
        print(prediction)
