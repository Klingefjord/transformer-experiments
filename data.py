import subprocess
import torch
import typing
from torch.utils.data import DataLoader, IterableDataset

from tokenizer import Tokenizer, create_tokenizer


def line_count(filename):
    """Efficiently count the number of lines in a file"""
    return int(subprocess.check_output(["wc", "-l", filename]).split()[0])


class TextDataset(IterableDataset):
    def __init__(
        self, path: str, encoder: Tokenizer, seq_len: int = 512, fractions=(0.0, 1.0)
    ) -> None:
        self.fractions = fractions
        self.seq_len = seq_len
        self.encoder = encoder
        self.path = path

    def read_file(self) -> typing.Generator:
        """Reads the file and yields each character."""
        lines = line_count(self.path)

        start_line, end_line = 0, 0

        if self.fractions[0] > 0.0:
            start = int(lines * self.start_fraction)
        if self.fractions[1] < 1.0:
            end = int(lines * self.end_fraction)

        with open(self.path, "r") as f:
            for i, line in enumerate(f):
                # start from the right line
                if i < start_line:
                    continue

                yield from line.strip("\n")

                # end at the right line
                if i > end_line:
                    break

    def __iter__(self) -> typing.Generator:
        """
        Stream characters from file and encode them.
        When the sequence reaches the desired length + 1
        (since the input to the transformer is shifted by one),
        yield the sequence.
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
    encoder = create_tokenizer("./data/pg16457.txt")
    return len(encoder.vocab)


def prepare_data(
    batch_size=128, seq_len=512, path="./data/dostoyevsky.txt"
) -> typing.Tuple[DataLoader, DataLoader, int]:
    """Create the test and validation dataloaders"""

    bpe_path = path.replace(".txt", ".bpe")
    vocab_path = path.replace(".txt", ".vocab")
    tokenizer = create_tokenizer(vocab_path, bpe_path)

    train_dataset = TextDataset(path, tokenizer, seq_len=seq_len, fractions=(0.0, 0.7))
    test_dataset = TextDataset(path, tokenizer, seq_len=seq_len, fractions=(0.7, 1.0))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=lambda x: torch.tensor(x)
    )

    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=lambda x: torch.tensor(x)
    )

    return train_loader, val_loader, len(tokenizer.vocab)
