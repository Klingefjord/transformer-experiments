import collections
import re
import typing


class VocabBuilder:
    """Class that builds a vocab from a text corpus using byte-pair encodings"""

    def __call__(self, path: str, num_merges: int = 10000) -> typing.Tuple[dict, dict]:
        """Build a vocab from a file"""
        vocab = self.load_vocab(path)
        tokens, merges = self.bpe(vocab, num_merges)
        return tokens, merges

    def get_pairs(self, vocab) -> dict:
        """Count pairs"""

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

    def load_vocab(self, path) -> dict:
        """Build a vocab from a file"""

        vocab = collections.defaultdict(int)
        with open(path) as f:
            for line in f:
                for word in line.split():
                    vocab[" ".join([c for c in word]) + " *"] += 1

            return vocab

    def bpe(self, vocab, num_merges=1000) -> typing.Tuple[dict, dict]:
        """Create byte-pair encodings from a text corpus"""

        merges = []

        for i in range(num_merges):
            pairs = self.get_pairs(vocab)

            if not pairs:
                break

            pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(pair, vocab)

            merges.append(pair)

            if i % (num_merges // 10) == 0:
                print(
                    "Iteration: {}\t Tokens: {}".format(i, len(self.get_tokens(vocab)))
                )

        # create a dict of the final tokens and return it with the merges.
        tokens = self.get_tokens(vocab)
        return tokens, merges
