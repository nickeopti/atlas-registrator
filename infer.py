from argparse import ArgumentParser

import torch
import skimage.exposure

import data
from model import FullRegressor

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--atlas', type=str)

    args = parser.parse_args()

    dataset = data.TiffTestImages(args.data)
    atlas = data.CCFv3(args.atlas, dorsoventral_rotation=(-1, 1), mediolateral_rotation=(-1, 1))
    model = FullRegressor.load_from_checkpoint(args.model)

    with torch.no_grad():
        predictions = []
        for image, label in dataset:
            fixed = atlas[label * 10][0]
            image = torch.from_numpy(
                skimage.exposure.equalize_adapthist(
                    image.numpy(), nbins=128, clip_limit=0.05
                )
            )
            prediction = model((fixed.unsqueeze(0), image.unsqueeze(0)))
            # prediction = model(torch.stack((fixed, image.expand(3, *image.shape))).unsqueeze(0))
            predictions.append(([p[0] for p in prediction], label))
    
    # for (z, d, m), label in predictions:
    #     print(round(z / 10), round(abs(z / 10 - label)), round(abs(d), 1), round(abs(m), 1), label)
    for (x, y, r), _ in predictions:
        print(x, y, r)

        print()
