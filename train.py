"""Train the model specified in the arguments and log results to wandb."""

import pytorch_lightning as pl
import torch
from data import prepare_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
from encoder import create_encoder
from model import GPTTransformer
from utils import Config


def train(config: Config):
    """Train the model with the given configuration."""

    # prepare the dataset
    train_loader, validation_loader, vocab_size = prepare_data(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
    )

    # update the vocab size in the config
    config.vocab_size = vocab_size

    # set up wandb logger
    # wandb_logger = WandbLogger(
    #     name=config.model_name,
    #     project="transformer-experiments",
    #     config=vars(config),
    # )

    # set up the model
    model = GPTTransformer(config)

    # set up the trainer
    trainer = pl.Trainer(
        max_epochs=config.epoch,
        # logger=wandb_logger,
        accelerator="gpu",
        devices=1,
    )

    # train the model
    trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="transformer")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_embed", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    # create the config from the args
    config = Config(**vars(args))

    # train a model from the config
    train(config)
