import argparse
import json
import re
from utils import bytes_to_unicode
import regex as re


class Tokenizer:
    """Byte-level byte-pair encoding tokenizer."""

    def __init__(self, vocab: dict, merges: list) -> None:
        self.merges = dict(zip(merges, range(len(merges))))

        self.encoder = vocab  # dict of tokens to their indices
        self.decoder = {
            v: k for k, v in vocab.items()
        }  # dict of indices to their tokens

        # create a decoder to translate all unicode bytes to their int byte.
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # a regex for splitting words like 't, 're, etc. into separate tokens.
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.cache = {}

    def encode(self, text) -> list[int]:
        """Encode text into byte-pair encodings"""

        bpe_indices = []
        tokens = self.pat.findall(text)

        for token in tokens:
            # encode the token as a bytes (b'') object
            token_bytes = token.encode("utf-8")

            # translate all bytes to their unicode string representation and flatten
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)

            # perform all the applicable bpe merges according to self.bpe_ranks
            token_merged = self._bpe(token_translated).split(" ")

            # translate all bpe tokens to integers
            token_indices = [self.encoder[bpe_token] for bpe_token in token_merged]

            # extend our running list of all output integers
            bpe_indices.extend(token_indices)

        return bpe_indices

    def decode(self, bpe_idices) -> str:
        """Decode a list of byte-pair indices into text"""

        tokens_merges = [self.decoder[token] for token in bpe_idices]
        tokens_flat = "".join(tokens_merges)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        return tokens_bytes.decode("utf-8", errors="replace")

    def _get_bigrams(self, word) -> set[tuple[str, str]]:
        """Return a set of bigrams for a word"""

        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token) -> str:
        """Iteratively merge all bpe tokens up the tree for the token. A token here is a word or subword."""

        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self._get_bigrams(word)

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
                pairs = self._get_bigrams(word)

        # concat all words into a string delineated by spaces
        word = " ".join(word)

        # cache the result and return
        self.cache[token] = word
        return word


def create_tokenizer(vocab_path, bpe_path) -> Tokenizer:
    # prepare the vocabulary and byte pair encodings
    vocab, merges = {}, {}

    # prepare the vocab
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # prepare the byte pair encodings
    with open(bpe_path, "r") as f:
        merges = f.read().split("\n")
        merges = [(tuple(l)) for l in merges]

    # create the encoder
    return Tokenizer(vocab, merges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe_path", default="./data/dostoyevsky.bpe", type=str)
    parser.add_argument("--vocab_path", default="./data/dostoyevsky.vocab", type=str)
    args = parser.parse_args()

    encoder = create_tokenizer(args.vocab_path, args.bpe_path)

    # encode a sentence
    indices = encoder.encode("Hello, world! I am Oliver.")
    print(indices)

    # decode a sentence
    string = encoder.decode(encoder.encode("Hello, world! I am Oliver."))
    print(string)
