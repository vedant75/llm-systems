from typing import List

from .bpe_reference_helper import (
    pretokenize,
    get_most_frequent_pair,
    apply_merge
)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
):
    # create a vocab
    vocab = {
        i: bytes([i])
        for i in range(256)
    }
    next_id = 256

    for t in special_tokens:
        vocab[next_id] = t.encode('utf-8')
        next_id +=1

    if vocab_size < len(vocab):
        raise ValueError(
            "vocab_size must be at least "
            "256 + number of special tokens"
        )


    # Read + pre-tokenize corpus
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8")

    word_counts = pretokenize(
        text,
        special_tokens,
    )

    # BPE Merges

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:

        pair = get_most_frequent_pair(
            word_counts
        )

        if pair is None:
            break

        vocab[next_id] = (
            pair[0] + pair[1]
        )

        merges.append(pair)

        # Replace pair throughout all pre-tokens.
        word_counts = apply_merge(
            word_counts,
            pair,
        )

        next_id += 1

    return vocab, merges
    