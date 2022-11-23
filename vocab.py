import argparse
import collections
import json
import re
import typing
import regex as re
from utils import bytes_to_unicode


class VocabBuilder:
    """Class that builds a vocab from a text corpus using byte-pair encodings"""

    def __init__(self):
        self.byte_encoder = bytes_to_unicode()

        # a regex for splitting words like 't, 're, etc. into separate tokens.
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def __call__(self, path: str, num_merges: int = 10000) -> typing.Tuple[dict, dict]:
        """Build a vocab from a file"""
        vocab = self.initialize_vocab(path)
        tokens, merges = self.bpe(vocab, num_merges)
        return tokens, merges

    def get_pairs(self, vocab: dict) -> dict:
        """Count the pairs in the vocab"""

        pairs = collections.defaultdict(int)
        for word, _ in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += 1

        return pairs

    def get_tokens(self, vocab: dict) -> dict[str, int]:
        """Returns a finalized vocab in the form of a dict of token -> index"""

        tokens = collections.defaultdict(int)

        i = 0

        # add base tokens
        for _ in range(2**8):
            tokens[self.byte_encoder[i]] = i
            i += 1

        # add merge tokens
        for word, _ in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                if token not in tokens:
                    tokens[token] = i
                    i += 1

        return tokens

    def merge_vocab(self, pair: tuple, v_in: dict) -> dict:
        """Merge all occurrences of the most frequent pair"""

        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in v_in:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def initialize_vocab(self, path: str) -> dict:
        """Initialize a vocab from a file"""

        vocab = collections.defaultdict(int)
        with open(path) as f:
            for line in f:
                tokens = self.pat.findall(line)
                for token in tokens:
                    vocab[
                        " ".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                    ] += 1

            return vocab

    def bpe(self, vocab: dict, num_merges=1000) -> typing.Tuple[dict, dict]:
        """Create byte-pair encodings from a text corpus"""

        merges = []

        for i in range(num_merges):
            if i % (num_merges // 10) == 0:
                print(
                    "Iteration: {}\t Tokens: {}".format(i, len(self.get_tokens(vocab)))
            )

            pairs = self.get_pairs(vocab)

            if not pairs:
                break

            pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(pair, vocab)

            merges.append(pair)

        # create a dict of the final tokens and return it with the merges.
        tokens = self.get_tokens(vocab)
        return tokens, merges


if __name__ == "__main__":
    builder = VocabBuilder()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_merges", type=int, default=300)
    parser.add_argument("--path", type=str, default="./data/dostoyevsky.txt")
    args = parser.parse_args()

    bpe_path = args.path.replace(".txt", ".bpe")
    vocab_path = args.path.replace(".txt", ".bpe")

    # build bpe vocab
    print(f"Building vocabulary for {args.path}...")
    tokens, merges = builder(args.path, args.num_merges)

    # save tokens
    print("Saving tokens...")
    with open("./data/dostoyevsky.bpe", "w") as f:
        for merge in merges:
            f.write(" ".join(merge) + "\n")

    # save merges
    print("Saving merges...")
    with open("./data/dostoyevsky.vocab", "w") as f:
        f.write(json.dumps(tokens))

    print(
        f"Built vocabulary of {len(tokens)} tokens. Should be 256 + {args.num_merges} = {256 + args.num_merges}."
    )
