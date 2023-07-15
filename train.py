from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import data
from model import CascadeRegressor, FullRegressor

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--cascade_depth', type=int, default=1)
    parser = pl.Trainer.add_argparse_args(parser)

    logger = pl.loggers.CSVLogger('logs')

    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(
        args=args,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(
            monitor='train_loss',
            save_top_k=5,
            mode='min',
        )],
        # auto_lr_find=True,
    )
    torch.set_float32_matmul_precision('high')

    dataset = data.CCFv3(args.data)
    train_data_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    # model = FullRegressor(learning_rate=args.learning_rate)
    model = CascadeRegressor(n=args.cascade_depth, learning_rate=args.learning_rate)

    # trainer.tune(model, train_data_loader)
    trainer.fit(model, train_data_loader)
