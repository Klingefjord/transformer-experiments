import argparse
import collections
import json
import os
import re
import typing


class Encoder:
    """Encoder class to convert text to byte-pair encodings"""

    def __init__(self, vocab, merges) -> None:
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.merges = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def encode(self, text) -> list[int]:
        """Encode text into byte-pair encodings"""

        bpe_idx = []
        tokens = [w + "*" for w in text.split(" ")]

        for token in tokens:
            # perform all the applicable bpe merges according to self.bpe
            token_merged = self.bpe(token).split(" ")
            # translate all bpe tokens to integers
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # extend our running list of all output integers
            bpe_idx.extend(token_ix)

        return bpe_idx

    def decode(self, bpe_idx) -> str:
        """Decode a list of byte-pair encodings into text"""
        return "".join([self.decoder[idx] for idx in bpe_idx]).replace("*", " ").strip()

    def get_bigrams(self, word) -> set[tuple[str, str]]:
        """Return a set of bigrams for a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token) -> str:
        """Iteratively merge all bpe tokens up the tree for the token (a word)"""
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


class VocabBuilder:
    """Class that builds a vocab from a text corpus using byte-pair encodings"""

    def __call__(self, path: str, num_merges: int = 10000) -> typing.Tuple[dict, dict]:
        """Run the vocab builder"""

        vocab = self._create_vocab(path)
        return self._bpe(vocab, num_merges)

    def _get_pairs(self, vocab) -> dict:
        """Count pairs"""

        pairs = collections.defaultdict(int)
        for word, _ in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += 1

        return pairs

    def _get_tokens(self, vocab: dict) -> dict:
        """Get tokens from vocab"""

        tokens = collections.defaultdict(int)
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens[token] += freq
        return tokens

    def _merge_vocab(self, pair: tuple, v_in: dict) -> dict:
        """Merge all occurrences of the most frequent pair"""

        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in v_in:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def _create_vocab(self, path) -> dict:
        """Build a vocab from a file"""

        vocab = collections.defaultdict(int)
        with open(path) as f:
            for line in f:
                for word in line.split():
                    vocab[" ".join([c for c in word]) + " *"] += 1

            return vocab

    def _bpe(self, vocab, num_merges=500) -> typing.Tuple[dict, dict]:
        """Create byte-pair encodings from a text corpus"""

        merges = []

        for i in range(num_merges):
            pairs = self._get_pairs(vocab)

            if not pairs:
                break

            pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(pair, vocab)

            merges.append(pair)

            if i % 10 == 0:
                print(
                    "Iteration: {}\t Tokens: {}".format(i, len(self._get_tokens(vocab)))
                )

        # replace the number of occurences with a unique index
        final_vocab = {k: i for i, (k, _) in enumerate(self._get_tokens(vocab).items())}

        # return the final vocab and the merges
        return final_vocab, merges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./pg16457.txt", type=str)
    parser.add_argument("--num_merges", default=1000, type=int)
    args = parser.parse_args()

    # prepare the vocabulary and byte pair encodings
    vocab, merges = {}, {}

    if not os.path.exists("vocab.txt") or not os.path.exists("merges.txt"):
        vb = VocabBuilder()
        vocab, merges = vb(args.path, args.num_merges)

        with open("vocab.txt", "w") as f:
            f.write(json.dumps(vocab))
        with open("merges.txt", "w") as f:
            f.write(json.dumps(merges))
    else:
        vocab = json.loads(open("vocab.txt", "r").read())
        merges = json.loads(open("merges.txt", "r").read())

    merges = [(tuple(l)) for l in merges]

    # create the encoder
    encoder = Encoder(vocab, merges)

    # encode a sentence
    indices = encoder.encode("Hello, world! I am Oliver.")
    print(indices)

    # decode a sentence
    string = encoder.decode(encoder.encode("Hello, world! I am Oliver."))
    print(string)
