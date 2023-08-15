import os
import os.path
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn
import torchvision.models.resnet

import augmentation
import resnet
import transformation


def transform(data):
    m, x, y, r, s, a = data

    return transformation.affine(
        m,
        x=x.detach().cpu().item(),
        y=y.detach().cpu().item(),
        angle=r.detach().cpu().item(),
        scale=s.detach().cpu().item(),
        aspect=a.detach().cpu().item()
    )


class BaseModel(pl.LightningModule):
    def __init__(self, size: int = 64, save_images_every_n_epochs: int = 0, learning_rate: float = 1e-3):
        super().__init__()

        self.size = size

        self.learning_rate = learning_rate

        self.save_every = save_images_every_n_epochs
        self.save_images = False
        self._image_dir = None

    def configure_step(self, batch_idx: int):
        self.save_images = self.save_every != 0 and self.current_epoch % self.save_every == 0 and batch_idx == 0

    @property
    def image_dir(self):
        if self._image_dir is None:
            self._image_dir = os.path.join(self.logger.log_dir, 'figures')
            os.makedirs(self._image_dir, exist_ok=True)

        return self._image_dir

    def on_after_batch_transfer(self, batch: list[torch.Tensor], _: int) -> list[torch.Tensor]:
        fixed, moving, x, y, r, s, a = batch

        *_, h, w = fixed.shape
        if w > h:
            location_factor = self.size / w
        else:
            location_factor = self.size / h

        if self.training:
            fixed = augmentation.augment(
                fixed,
                augmentation.compose(
                    augmentation.ALL_AUGMENTATIONS,
                    probability_all_sampled=0.4
                )
            )

        fixed = fixed.unsqueeze(1)
        moving = moving.unsqueeze(1)

        fixed = transformation.normalise(fixed)
        moving = transformation.normalise(moving)

        fixed = transformation.pad_resize(fixed, self.size, self.size)
        moving = transformation.pad_resize(moving, self.size, self.size)

        return [
            fixed,
            moving,
            x * location_factor,
            y * location_factor,
            r,
            s,
            a
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 1e-8)

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
        }


class JointRegressor(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.resnet = torchvision.models.resnet.resnet50(
            weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2,
        )
        conv1_weights = self.resnet.conv1.weight.sum(dim=1, keepdim=True)
        self.resnet.conv1.weight = torch.nn.Parameter(conv1_weights)

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

        if self.save_images:
            moved = torch.stack(tuple(map(transform, zip(m, *z))))

            torchvision.utils.save_image(f, os.path.join(self.image_dir, f'{self.current_epoch}_fixed.png'))
            torchvision.utils.save_image(m, os.path.join(self.image_dir, f'{self.current_epoch}_moving.png'))
            torchvision.utils.save_image(moved, os.path.join(self.image_dir, f'{self.current_epoch}_moved.png'))

        return z

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        self.configure_step(batch_idx)

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


class Regressor(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.resnet = resnet.create_resnet(
            model_depth=10,
            pretrained=True,
            n_input_channels=2,
            num_classes=1024,
            activation_function=torch.nn.ReLU,
            do_avg_pool=False,
            input_size=(input_size, input_size),
        )
        self.elu = torch.nn.ELU()
        self.fc1 = torch.nn.Linear(1024, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.regressor = torch.nn.Linear(32, 5)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_ = torch.hstack(x)

        z = self.resnet(x_)
        z = self.elu(z)
        z = self.fc1(z)
        z = self.elu(z)
        z = self.fc2(z)
        z = self.elu(z)
        z = self.regressor(z)

        return z


class CascadeRegressor(BaseModel):
    def __init__(self, cascade_layers: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.layers = torch.nn.ModuleList(
            Regressor(input_size=kwargs['size'])
            for _ in range(cascade_layers)
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        fixed, moving = x

        if self.save_images:
            torchvision.utils.save_image(fixed, os.path.join(self.image_dir, f'{self.current_epoch}_fixed.png'))
            torchvision.utils.save_image(moving, os.path.join(self.image_dir, f'{self.current_epoch}_moving.png'))

        cum_x, cum_y, cum_r = (torch.zeros(moving.shape[0], device=moving.device) for _ in range(3))
        cum_s, cum_a = (torch.ones(moving.shape[0], device=moving.device) for _ in range(2))
        cum_x_abs, cum_y_abs, cum_r_abs = (torch.zeros(moving.shape[0], device=moving.device) for _ in range(3))
        cum_s_abs, cum_a_abs = (torch.ones(moving.shape[0], device=moving.device) for _ in range(2))

        moved = moving.clone()
        for i, layer in enumerate(self.layers):
            pred_x, pred_y, pred_r, pred_s, pred_a = map(torch.stack, zip(*layer((fixed, moved))))
            pred_s = torch.sigmoid(pred_s) * 2
            pred_a = torch.sigmoid(pred_a) * 2

            if not self.training:
                print(f'{pred_x=}, {pred_y=}, {pred_r}, {pred_s}, {pred_a}')

            cum_x += pred_x
            cum_y += pred_y
            cum_r += pred_r
            cum_s *= pred_s
            cum_a *= pred_a

            cum_x_abs += pred_x.abs()
            cum_y_abs += pred_y.abs()
            cum_r_abs += pred_r.abs()
            cum_s_abs *= (pred_s - 1).abs() + 1
            cum_a_abs *= (pred_a - 1).abs() + 1

            moved = torch.stack(tuple(map(transform, zip(moving, cum_x, cum_y, cum_r, cum_s, cum_a))))

            if self.save_images:
                torchvision.utils.save_image(moved, os.path.join(self.image_dir, f'{self.current_epoch}_moved_{i}.png'))

        return moving, cum_x, cum_y, cum_r, cum_s, cum_a, cum_x_abs, cum_y_abs, cum_r_abs, cum_s_abs, cum_a_abs

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        fixed, moving, offset_x, offset_y, rotation, scale, aspect = batch
        assert fixed.shape == moving.shape

        self.configure_step(batch_idx)

        _, pred_x, pred_y, pred_r, pred_s, pred_a, x_abs, y_abs, r_abs, s_abs, a_abs = self((fixed, moving))

        location_loss = ((offset_x - pred_x)**2 + (offset_y - pred_y)**2).mean()
        rotation_loss = ((rotation - pred_r)**2).mean()
        scale_loss = ((scale - pred_s)**2).mean() + ((aspect - pred_a)**2).mean()

        smoothness_regularisor = (
            ((pred_x.abs() - x_abs)**2).mean() +
            ((pred_y.abs() - y_abs)**2).mean() +
            ((pred_r.abs() - r_abs)**2).mean() +
            (((pred_s - 1).abs() + 1 - s_abs)**2).mean() +
            (((pred_a - 1).abs() + 1 - a_abs)**2).mean()
        )

        loss = location_loss + rotation_loss + 100 * scale_loss + smoothness_regularisor

        self.log('train_loss', loss)
        self.log('location_loss', location_loss)
        self.log('rotation_loss', rotation_loss)
        self.log('scale_loss', scale_loss)
        self.log('smoothness', smoothness_regularisor.mean())
        self.log('s_x', ((pred_x.abs() - x_abs)**2).mean())
        self.log('s_y', ((pred_y.abs() - y_abs)**2).mean())
        self.log('s_r', ((pred_r.abs() - r_abs)**2).mean())
        self.log('s_s', (((pred_s - 1).abs() + 1 - s_abs)**2).mean())
        self.log('s_a', (((pred_a - 1).abs() + 1 - a_abs)**2).mean())
        self.log('mae_x', (offset_x - pred_x).abs().mean())
        self.log('mae_y', (offset_y - pred_y).abs().mean())
        self.log('mae_r', (rotation - pred_r).abs().mean())
        self.log('mae_s', (scale - pred_s).abs().mean())
        self.log('mae_a', (aspect - pred_a).abs().mean())

        return loss


class GridRegressor(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = Regressor(input_size=kwargs['size'])

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        fixed, moving = x

        if self.save_images:
            torchvision.utils.save_image(fixed, os.path.join(self.image_dir, f'{self.current_epoch}_fixed.png'))
            torchvision.utils.save_image(moving, os.path.join(self.image_dir, f'{self.current_epoch}_moving.png'))

        pred_x, pred_y, pred_r, pred_s, pred_a = map(torch.stack, zip(*self.model((fixed, moving))))
        pred_s = torch.sigmoid(pred_s) * 2
        pred_a = torch.sigmoid(pred_a) * 2

        if self.save_images:
            moved = torch.stack(tuple(map(transform, zip(moving, pred_x, pred_y, pred_r, pred_s, pred_a))))
            torchvision.utils.save_image(moved, os.path.join(self.image_dir, f'{self.current_epoch}_moved.png'))

        return pred_x, pred_y, pred_r, pred_s, pred_a

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        fixed, moving, *transforms = batch
        assert fixed.shape == moving.shape

        self.configure_step(batch_idx)

        predictions = self((fixed, moving))

        *_, h, w = fixed.shape
        grid_xx, grid_yy = torch.meshgrid(
            torch.linspace(0, self.size - 1, steps=10),
            torch.linspace(0, self.size - 1, steps=10),
            indexing='ij'
        )
        points = torch.vstack((
            grid_xx.flatten(),
            grid_yy.flatten(),
            torch.ones(grid_xx.numel())
        ))

        loss = torch.zeros(1, device=self.device)
        for (x_true, y_true, r_true, s_true, a_true), (x, y, r, s, a) in zip(zip(*transforms), zip(*predictions)):
            transformation_matrix_target = transformation.affine_transformation_matrix(
                x=x_true,
                y=y_true,
                angle=r_true,
                scale=s_true,
                aspect=a_true,
                width=w,
                height=h
            )
            transformation_matrix_prediction = transformation.affine_transformation_matrix(
                x=x,
                y=y,
                angle=r,
                scale=s,
                aspect=a,
                width=w,
                height=h
            )

            xs, ys, _ = transformation_matrix_target @ points - transformation_matrix_prediction @ points
            loss += (xs**2 + ys**2).mean()
        loss /= fixed.shape[0]  # mean over batch

        self.log('train_loss', loss)

        return loss


class Classifier(BaseModel):
    def __init__(self, layered: bool = True, tiny: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.layered = layered
        self.tiny = tiny

        if self.tiny:
            self.resnet = resnet.create_resnet(
                model_depth=10,
                n_input_channels=2 if self.layered else 1,
                num_classes=256,
                activation_function=torch.nn.ELU,
                linear_factor=16,
            )
        else:
            self.resnet = torchvision.models.resnet.resnet50(
                weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2,
            )

            conv1_weights = self.resnet.conv1.weight.sum(dim=1, keepdim=True)
            if self.layered:
                conv1_weights = conv1_weights.repeat(1, 2, 1, 1)
            self.resnet.conv1.weight = torch.nn.Parameter(conv1_weights)

        self.elu = torch.nn.ELU()
        if self.tiny:
            self.hidden = torch.nn.Linear(256 if self.layered else 512, 128)
        else:
            self.hidden = torch.nn.Linear(1000 if self.layered else 2000, 128)
        self.horizontal = torch.nn.Linear(128, 3)
        self.vertical = torch.nn.Linear(128, 3)
        self.rotation = torch.nn.Linear(128, 3)
        self.scale = torch.nn.Linear(128, 3)
        self.aspect = torch.nn.Linear(128, 3)

        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        fixed, moving = x

        if self.layered:
            z = self.resnet(torch.hstack((fixed, moving)))
        else:
            f = self.resnet(fixed)
            m = self.resnet(moving)
            z = torch.hstack((f, m))

        z = self.elu(z)
        z = self.hidden(z)
        z = self.elu(z)

        h = self.horizontal(z)
        v = self.vertical(z)
        r = self.rotation(z)
        s = self.scale(z)
        a = self.aspect(z)

        return h, v, r, s, a

    def _dichotomise(self, x: torch.Tensor, threshold: float = 0, epsilon: float = 5):
        y = torch.zeros_like(x, dtype=torch.long, device=x.device)
        y[x > threshold + epsilon] = 0
        y[x < threshold - epsilon] = 1
        y[torch.logical_and(x <= threshold + epsilon, x >= threshold - epsilon)] = 2

        return y

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        fixed, moving, offset_x, offset_y, rotation, scale, aspect = batch

        self.configure_step(batch_idx)
        if self.save_images:
            torchvision.utils.save_image(fixed, os.path.join(self.image_dir, f'{self.current_epoch}_fixed.png'))
            torchvision.utils.save_image(moving, os.path.join(self.image_dir, f'{self.current_epoch}_moving.png'))

        moved = moving.clone()
        cum_x, cum_y, cum_r = (torch.zeros(moving.shape[0], device=moving.device) for _ in range(3))
        cum_s, cum_a = (torch.ones(moving.shape[0], device=moving.device) for _ in range(2))

        loss = torch.zeros(fixed.shape[0], device=fixed.device)
        for i in range(25):
            pred_x, pred_y, pred_r, pred_s, pred_a = self((fixed, moved))

            loss_h = self.loss_function(pred_x, self._dichotomise(offset_x - cum_x, 0, 3))
            loss_v = self.loss_function(pred_y, self._dichotomise(offset_y - cum_y, 0, 3))
            loss_r = self.loss_function(pred_r, self._dichotomise(rotation - cum_r, 0, 1))
            loss_s = self.loss_function(pred_s, self._dichotomise(scale - cum_s, 1, 0.05))
            loss_a = self.loss_function(pred_a, self._dichotomise(aspect - cum_a, 1, 0.05))

            location_loss = offset_x**2 * loss_h + offset_y**2 * loss_v
            rotation_loss = rotation**2 * loss_r
            scale_loss = (torch.abs(scale - 1) * 10)**2 * loss_s + (torch.abs(aspect - 1) * 10)**2 * loss_a

            loss += location_loss + rotation_loss + scale_loss

            cum_x += (offset_x - cum_x) / 5
            cum_y += (offset_x - cum_y) / 5
            cum_r += (rotation - cum_r) / 5
            cum_s *= 1 + (scale - cum_s) / 5
            cum_a *= 1 + (aspect - cum_a) / 5

            moved = torch.stack(tuple(map(transform, zip(moving, cum_x, cum_y, cum_r, cum_s, cum_a))))

            if self.save_images:
                torchvision.utils.save_image(moved, os.path.join(self.image_dir, f'{self.current_epoch}_moved_{i}.png'))

        self.log('train_loss', loss.mean())
        self.log('location_loss', location_loss.mean())
        self.log('rotation_loss', rotation_loss.mean())
        self.log('scale_loss', scale_loss.mean())

        return loss.mean()
