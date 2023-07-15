from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn
import torchvision.models.resnet

import augmentation
import transformation


class FullRegressor(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        self.resnet = torchvision.models.resnet.resnet50(
            weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2,
        )
        self.elu = torch.nn.ELU()
        self.hidden = torch.nn.Linear(2000, 128)
        self.regressor = torch.nn.Linear(128, 5)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        f, m = x
        f_ = self.resnet(f)
        m_ = self.resnet(m)

        z = torch.hstack((f_, m_))
        z = self.elu(z)
        z = self.hidden(z)
        z = self.elu(z)
        z = self.regressor(z)

        return z

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        fixed, moving, offset_x, offset_y, rotation, scale, aspect = batch

        pred_x, pred_y, pred_r, pred_s, pred_a = map(torch.stack, zip(*self((fixed, moving))))

        location_loss = ((offset_x - pred_x)**2 + (offset_y - pred_y)**2).mean()
        rotation_loss = ((rotation - pred_r)**2).mean()
        scale_loss = ((scale - pred_s)**2).mean() + ((aspect - pred_a)**2).mean()

        loss = location_loss + rotation_loss + scale_loss

        self.log('train_loss', loss)
        self.log('location_loss', location_loss)
        self.log('rotation_loss', rotation_loss)
        self.log('scale_loss', scale_loss)
        self.log('mae_x', (offset_x - pred_x).abs().mean())
        self.log('mae_y', (offset_y - pred_y).abs().mean())
        self.log('mae_r', (rotation - pred_r).abs().mean())
        self.log('mae_s', (scale - pred_s).abs().mean())
        self.log('mae_a', (aspect - pred_a).abs().mean())

        return loss
    
    def on_after_batch_transfer(self, batch: list[torch.Tensor], batch_idx) -> list[torch.Tensor]:
        a, b, *c = batch

        b_ = augmentation.augment(
            b,
            augmentation.compose(
                augmentation.ALL_AUGMENTATIONS,
                probability_all_sampled=0.4
            )
        )

        return [
            a.unsqueeze(1).expand(-1, 3, -1, -1),
            b_.unsqueeze(1).expand(-1, 3, -1, -1),
            *c
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 1e-10)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


def transform(data):
    m, x, y, r, s, a = data

    return transformation.affine(
        m,
        angle=r,
        translate=(x, y),
        scale=(s, s * a),
        shear=(0, 0)
    )



class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet.resnet50(
            weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2,
        )
        self.elu = torch.nn.ELU()
        self.hidden = torch.nn.Linear(2000, 128)
        self.regressor = torch.nn.Linear(128, 5)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        f, m = x
        f_ = self.resnet(f)
        m_ = self.resnet(m)

        z = torch.hstack((f_, m_))
        z = self.elu(z)
        z = self.hidden(z)
        z = self.elu(z)
        z = self.regressor(z)

        return z


class CascadeRegressor(pl.LightningModule):
    def __init__(self, n: int = 1, learning_rate: float = 1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        self.layers = torch.nn.ModuleList(
            Regressor()
            for _ in range(n)
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        fixed, moving = x

        fixed = transformation.clahe(fixed)
        moving = transformation.clahe(moving)

        cum_x, cum_y, cum_r, cum_s, cum_a = (torch.zeros(moving.shape[0], device=moving.device) for _ in range(5))

        for layer in self.layers:
            pred_x, pred_y, pred_r, pred_s, pred_a = map(torch.stack, zip(*layer((fixed, moving))))

            if not self.training:
                print(f'{pred_x=}, {pred_y=}, {pred_r}, {pred_s}, {pred_a}')

            cum_x += pred_x
            cum_y += pred_y
            cum_r += pred_r
            cum_s *= pred_s
            cum_a *= pred_a

            moving = torch.stack(tuple(map(transform, zip(moving, pred_x, pred_y, pred_r, pred_s, pred_a))))

        return moving, cum_x, cum_y, cum_r, cum_s, cum_a

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        fixed, moving, offset_x, offset_y, rotation, scale, aspect = batch

        _, pred_x, pred_y, pred_r, pred_s, pred_a = self((fixed, moving))

        location_loss = ((offset_x - pred_x)**2 + (offset_y - pred_y)**2).mean()
        rotation_loss = ((10 * (rotation - pred_r))**2).mean()
        scale_loss = ((100 * (scale - pred_s))**2).mean() + ((100 * (aspect - pred_a))**2).mean()

        loss = location_loss + rotation_loss + 10**4 * scale_loss

        self.log('train_loss', loss)
        self.log('location_loss', location_loss)
        self.log('rotation_loss', rotation_loss)
        self.log('scale_loss', scale_loss)
        self.log('mae_x', (offset_x - pred_x).abs().mean())
        self.log('mae_y', (offset_y - pred_y).abs().mean())
        self.log('mae_r', (rotation - pred_r).abs().mean())
        self.log('mae_s', (scale - pred_s).abs().mean())
        self.log('mae_a', (aspect - pred_a).abs().mean())

        return loss

    def on_after_batch_transfer(self, batch: list[torch.Tensor], batch_idx) -> list[torch.Tensor]:
        a, b, *c = batch

        a_ = augmentation.augment(
            a,
            augmentation.compose(
                augmentation.ALL_AUGMENTATIONS,
                probability_all_sampled=0.4
            )
        )

        return [
            a_.unsqueeze(1).expand(-1, 3, -1, -1),
            b.unsqueeze(1).expand(-1, 3, -1, -1),
            *c
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 1e-10)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
