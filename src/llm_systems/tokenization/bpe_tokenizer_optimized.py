from __future__ import annotations

import os

from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from .bpe_optimized_helper import  (
    find_chunk_boundaries, 
    pretokenize,
    build_pair_index,
    max_pair,
    apply_incremental_merge
)



def train_optimized_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 4,
):
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    next_id = 256

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    if vocab_size < len(vocab):
        raise ValueError(
            "vocab_size must be at least "
            "256 + number of special tokens"
        )

    # Safe chunk boundaries

    if special_tokens:
        boundary_token = special_tokens[0].encode("utf-8")

        with open(input_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=num_workers,
                split_special_token=boundary_token,
            )
    else:
        file_size = os.path.getsize(input_path)
        chunk_boundaries = [0, file_size]

    # Parallel pretokenization

    combined_counter = Counter()

    with ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures = []

        for start_byte, end_byte in zip(
            chunk_boundaries[:-1],
            chunk_boundaries[1:],
        ):
            future = executor.submit(
                pretokenize,
                input_path=input_path,
                start_byte=start_byte,
                end_byte=end_byte,
                special_tokens=special_tokens,
            )

            futures.append(future)

        for future in futures:
            combined_counter.update(
                future.result()
            )

    # Incremental BPE
    (
        words,
        word_freqs,
        pair_counts,
        pair_to_words,
    ) = build_pair_index(combined_counter)

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        winner = max_pair(pair_counts)

        merges.append(winner)

        vocab[len(vocab)] = (
            winner[0] + winner[1]
        )

        apply_incremental_merge(
            winner=winner,
            words=words,
            word_freqs=word_freqs,
            pair_counts=pair_counts,
            pair_to_words=pair_to_words,
        )

    return vocab, merges