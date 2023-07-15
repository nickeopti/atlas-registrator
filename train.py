from argparse import ArgumentParser

import pytorch_lightning as pl
import selector
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import data
import model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    dataset = selector.add_arguments(parser, 'dataset', data.CCFv3Torch)()
    train_data_loader = selector.add_arguments(parser, 'dataloader', DataLoader)(
        dataset=dataset,
        num_workers=0,
    )
    network = selector.add_options_from_module(parser, 'model', model, pl.LightningModule)()

    logger = pl.loggers.CSVLogger('logs')
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args=parser,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                monitor='train_loss',
                save_top_k=5,
                mode='min',
            ),
        ],
    )

    trainer.fit(network, train_data_loader)
