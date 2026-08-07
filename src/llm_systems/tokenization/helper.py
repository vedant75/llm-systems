from __future__ import annotations

from collections import Counter

import regex as re


PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(
    text: str,
    special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:

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
        for match in re.finditer(PAT, document):
            word_bytes = match.group().encode("utf-8")

            symbols = tuple(
                bytes([b])
                for b in word_bytes
            )

            word_counts[symbols] += 1

    return word_counts


# pair_counting
def get_most_frequent_pair(
    word_counts: Counter[tuple[bytes, ...]],
) -> tuple[bytes, bytes] | None:

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    for word, frequency in word_counts.items():
        for pair in zip(
            word,
            word[1:],
        ):
            pair_counts[pair] += frequency

    if not pair_counts:
        return None

    return max(
        pair_counts,
        key=lambda pair: (
            pair_counts[pair],
            pair,
        ),
    )

# merge a pair
def apply_merge(
    word_counts: Counter[tuple[bytes, ...]],
    pair: tuple[bytes, bytes],
) -> Counter[tuple[bytes, ...]]:

    left, right = pair
    merged = left + right

    new_word_counts: Counter[tuple[bytes, ...]] = Counter()

    for word, frequency in word_counts.items():
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
        new_word_counts[tuple(new_word)] += frequency

    return new_word_counts