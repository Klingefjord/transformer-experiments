"""Train the model specified in the arguments and log results to wandb."""

import pytorch_lightning as pl
import torch
from data import prepare_data
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse
from model import GPTTransformer
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
    wandb_logger = WandbLogger(
        name=config.model_name,
        project="dostoyevsky",
        config=vars(config),
    )

    # set up the model
    model = GPTTransformer(config)

    # set up the trainer
    trainer = pl.Trainer(
        max_epochs=config.epoch,
        logger=wandb_logger,
        accelerator="gpu",
        gradient_clip_val=1.0,
        devices=1,
    )

    # train the model
    trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()

    parser.add_argument("--model_name", default="gpt-mini")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=4e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    # only applied to linear layers
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_embed", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_norm_clip", type=float, default=1.0)
    # should we intersperse the training with printing out samples from the model?
    parser.add_argument("--should_sample", type=bool, default=True)

    args = parser.parse_args()

    # create the config from the args
    config = Config(**vars(args))

    # train a model from the config
    train(config)
