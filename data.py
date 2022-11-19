import torch
import typing
from torch.utils.data import DataLoader, IterableDataset

from encoder import Encoder, create_encoder


class TextDataset(IterableDataset):
    def __init__(self, path: str, encoder: Encoder, seq_len: int = 512) -> None:
        self.seq_len = seq_len
        self.encoder = encoder
        self.path = path

    def read_file(self) -> typing.Generator:
        """Reads the file and yields each character."""
        with open(self.path, "r") as f:
            for line in f:
                yield from line.strip("\n")

    def __iter__(self) -> typing.Generator:
        """
        Stream characters from file and encode them.
        When the sequence reaches the desired length + 1
        (since the input is shifted by one), yield the sequence.
        """
        sequence = []
        for char in self.read_file():
            tokens = self.encoder.encode(char)
            for token in tokens:
                if len(sequence) == self.seq_len + 1:
                    yield sequence
                    sequence = []
                sequence.append(token)


def get_vocab_size():
    """Returns the size of the vocabulary"""
    encoder = create_encoder("./data/pg16457.txt")
    return len(encoder.encoder)


def prepare_data(
    batch_size=128,
    seq_len=512,
    train_path="./data/gutenberg_train.txt",
    test_path="./data/gutenberg_test.txt",
) -> typing.Tuple[DataLoader, DataLoader, int]:
    """Create the test and validation dataloaders"""

    encoder = create_encoder("./data/pg16457.txt")
    train_dataset = TextDataset(train_path, encoder, seq_len=seq_len)
    test_dataset = TextDataset(test_path, encoder, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=lambda x: torch.tensor(x)
    )

    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=lambda x: torch.tensor(x)
    )

    return train_loader, val_loader, len(encoder.encoder)
