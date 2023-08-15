from argparse import ArgumentParser

import torch
import torchvision

import data
import transformation
from model import Classifier

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--layered', type=bool)
    parser.add_argument('--tiny', type=bool)
    parser.add_argument('--size', type=int)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--atlas', type=str)
    parser.add_argument('--slide', type=int)
    args = parser.parse_args()

    atlas = data.CCFv3Torch(args.atlas, dorsoventral_rotation=(-0.0, 0.0), mediolateral_rotation=(-0.0, 0.0), accelerator=args.device)
    model = Classifier.load_from_checkpoint(args.model, layered=args.layered, size=args.size).to(args.device)
    model.eval()

    with torch.no_grad():
        fixed, moving, *c = atlas[args.slide]
        fixed, moving, *c = model.on_after_batch_transfer(
            [fixed.unsqueeze(0), moving.unsqueeze(0), *c], 0
        )

        torchvision.utils.save_image(fixed, 'cp/fixed.png')
        torchvision.utils.save_image(moving, 'cp/moving.png')

        moved = moving.clone()
        cum_h, cum_v, cum_r = (0, 0, 0)
        cum_s, cum_a = (1, 1)

        for i in range(500):
            predictions = torch.stack(model((fixed, moved))).squeeze()
            h, v, r, s, a = predictions

            actions, = torch.where(predictions.argmax(dim=1) != 2)
            if len(actions) == 0:
                break

            action = actions[torch.randint(actions.numel(), (1,))].item()
            match action:
                case 0:  # horizontal translation
                    if h.argmax() == 0:
                        cum_h += 5
                    else:
                        cum_h -= 5
                case 1:  # vertical translation
                    if v.argmax() == 0:
                        cum_v += 5
                    else:
                        cum_v -= 5
                case 2:  # rotation
                    if r.argmax() == 0:
                        cum_r += 1
                    else:
                        cum_r -= 1
                case 3:  # scale
                    if s.argmax() == 0:
                        cum_s *= 1.05
                    else:
                        cum_s /= 1.05
                case 4:  # aspect
                    if a.argmax() == 0:
                        cum_a *= 1.05
                    else:
                        cum_a /= 1.05

            moved = transformation.affine(
                moving,
                x=cum_h,
                y=cum_v,
                angle=cum_r,
                scale=cum_s,
                aspect=cum_a
            )

            torchvision.utils.save_image(moved, f'cp/moved_{i}.png')
