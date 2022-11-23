import torch
import argparse
import collections
import json
import os
import re
import typing
from vocab import VocabBuilder
from utils import bytes_to_unicode

class Encoder:
    """Encoder class to convert text to byte-pair encodings"""

    def __init__(self, vocab, merges) -> None:
        self.vocab = vocab
        self.itos = {v: k for k, v in vocab.items()}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.merges = dict(zip(merges, range(len(merges))))
        # a regex for splitting words like 't, 're, etc. into separate tokens.
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.cache = {}

    def encode(self, text) -> list[int]:
        """Encode text into byte-pair encodings"""

        bpe_idx = []
        tokens = [w + "*" for w in text.split(" ")]

        for token in tokens:
            # encode the token as a bytes (b'') object
            token_bytes = token.encode("utf-8")

            # translate all bytes to their unicode string representation and flatten
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)

            token_merged = self.bpe(token).split(" ")
            token_ix = [self.vocab[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)

        return bpe_idx

    def decode(self, bpe_idx) -> str:
        """Decode a list of byte-pair encodings into text"""
        return "".join([self.itos[idx] for idx in bpe_idx]).replace("*", " ").strip()

    def get_bigrams(self, word) -> set[tuple[str, str]]:
        """Return a set of bigrams for a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token) -> str:
        """Iteratively merge all bpe tokens up the tree for the token. A token here is a word or subword."""

        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_bigrams(word)

        if not pairs:
            return token

        while True:
            # lowest mergable bigram
            bigram = min(pairs, key=lambda p: self.merges.get(p, float("inf")))

            if bigram not in self.merges:
                break

            first, second = bigram

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            new_word = []
            i = 0

            while i < len(word):

                # look for an occurence of the first character of the bigram
                # if we don't find it, append remaining chars and break the loop
                try:
                    j = word.index(first, i)
                    new_word += word[i:j]
                    i = j
                except:

                    new_word += word[i:]
                    break

                # We found the first character of the bigram, now look for the second
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            # continue the loop with new pairs
            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = self.get_bigrams(word)

        # concat all words into a string delineated by spaces
        word = " ".join(word)

        # cache the result and return
        self.cache[token] = word
        return word


class Tokenizer:
    """PyTorch-aware class that wraps the Encoder above"""

    def __init__(self, vocab, merges) -> None:
        self.encoder = create_encoder(vocab, merges)

    def __call__(self, text, return_tensors="pt"):
        # PyTorch only; here because we want to match huggingface/transformers interface
        assert return_tensors == "pt"
        # single string input for now, in the future potentially a list of strings
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1
        idx = [self.encoder.encode(text)]
        # wrap into PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # decode indices to text
        text = self.encoder.decode(idx.tolist())
        return text


def create_encoder(path, num_merges=1000):
    # prepare the vocabulary and byte pair encodings
    vocab, merges = {}, {}
    if not os.path.exists("vocab.txt") or not os.path.exists("merges.txt"):
        vocab, merges = VocabBuilder()(path, num_merges)

        with open("vocab.txt", "w") as f:
            f.write(json.dumps(vocab))
        with open("merges.txt", "w") as f:
            f.write(json.dumps(merges))
    else:
        with open("vocab.txt") as f:
            vocab = json.loads(f.read())
        with open("merges.txt") as f:
            merges = json.loads(f.read())
    merges = [(tuple(l)) for l in merges]

    return Encoder(vocab, merges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./data/pg16457.txt", type=str)
    parser.add_argument("--num_merges", default=1000, type=int)
    args = parser.parse_args()

    encoder = create_encoder(args.path, args.num_merges)

    # encode a sentence
    indices = encoder.encode("Hello, world! I am Oliver.")
    print(indices)

    # decode a sentence
    string = encoder.decode(encoder.encode("Hello, world! I am Oliver."))
    print(string)
