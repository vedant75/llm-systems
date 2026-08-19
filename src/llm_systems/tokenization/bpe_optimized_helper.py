from __future__ import annotations

from collections import (
    Counter,
    defaultdict
)
from typing import BinaryIO

import regex as re
import os

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize_text(
    text: str,
) -> list[tuple[bytes, ...]]:
    pretokens = []

    for match in re.finditer(PAT, text):
        word_bytes = match.group().encode("utf-8")

        symbols = tuple(
            bytes([b])
            for b in word_bytes
        )

        pretokens.append(symbols)
    
    return pretokens


def pretokenize(
    input_path: str,
    start_byte: int,
    end_byte: int,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:

    with open(input_path, 'rb') as f:
        f.seek(start_byte)
        bytes_to_read = end_byte - start_byte
        chunk = f.read(bytes_to_read)

    text = chunk.decode("utf-8")

    if special_tokens:
        special_pattern = "|".join(
            re.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )

        documents = re.split(
            special_pattern,
            text,
        )
    else:
        documents = [text]

    word_counts: Counter[tuple[bytes, ...]] = Counter()

    for document in documents:
        pretokens = pretokenize_text(document)

        for pretoken in pretokens:
            word_counts[pretoken] += 1

    return word_counts


# Optimizing merges

def build_pair_index(word_counts):
    words = {}
    word_freqs = {}
    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    for word_id, (word, freq) in enumerate(word_counts.items()):
        words[word_id] = word
        word_freqs[word_id] = freq

        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_id)

    return (
        words,
        word_freqs,
        pair_counts,
        pair_to_words,
    )

def get_local_pair_counts(word):
    pair_counts = Counter()

    for pair in zip(word, word[1:]):
        pair_counts[pair] += 1

    return pair_counts

def merge_word(
    word: tuple[bytes, ...],
    target_pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:

    left, right = target_pair
    merged = left + right

    new_word: list[bytes] = []
    i = 0
    while i < len(word):
        if (
            i + 1 < len(word)
            and word[i] == left
            and word[i + 1] == right
        ):
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1


    return tuple(new_word)

def max_pair(pair_count):
  return max(
      pair_count,
      key=lambda pair: (
        pair_count[pair],
        pair,
    ),
  )

def apply_incremental_merge(
    winner,
    words,
    word_freqs,
    pair_counts,
    pair_to_words,
):
    affected_word_ids = pair_to_words[winner].copy()

    for word_id in affected_word_ids:
        old_word = words[word_id]
        freq = word_freqs[word_id]

        # Remove old representation
        old_local_counts = get_local_pair_counts(old_word)

        for pair, local_count in old_local_counts.items():
            pair_counts[pair] -= local_count * freq

            assert pair_counts[pair] >= 0
            assert word_id in pair_to_words[pair]

            pair_to_words[pair].discard(word_id)

            if pair_counts[pair] == 0:
                del pair_counts[pair]

            if not pair_to_words[pair]:
                del pair_to_words[pair]

        # Apply BPE merge
        new_word = merge_word(old_word, winner)
        words[word_id] = new_word

        # Add new representation
        new_local_counts = get_local_pair_counts(new_word)

        for pair, local_count in new_local_counts.items():
            pair_counts[pair] += local_count * freq
            pair_to_words[pair].add(word_id)


# Tokenizer Helper
def split_on_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[tuple[str, bool]]:

    if not special_tokens:
        return [(text, False)]

    # Longer special tokens first in case one token
    # is a prefix/subsequence of another.
    sorted_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    special_pattern = (
        "("
        + "|".join(
            re.escape(token)
            for token in sorted_tokens
        )
        + ")"
    )

    parts = re.split(
        special_pattern,
        text,
    )

    special_token_set = set(special_tokens)

    result = []

    for part in parts:
        if part == "":
            continue

        result.append(
            (
                part,
                part in special_token_set,
            )
        )

    return result