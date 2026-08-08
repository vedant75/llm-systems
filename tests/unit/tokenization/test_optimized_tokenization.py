from collections import Counter

import pytest

from llm_systems.tokenization.bpe_optimized_helper import (
    build_pair_index,
    get_local_pair_counts,
    merge_word,
    max_pair,
    apply_incremental_merge,
)

from llm_systems.tokenization.bpe_tokenizer_optimized import (
    train_optimized_bpe,
)

# test Pair Counting
def test_get_local_pair_counts():
    word = (b"A", b"B", b"C")

    result = get_local_pair_counts(word)

    assert result == Counter({
        (b"A", b"B"): 1,
        (b"B", b"C"): 1,
    })


def test_get_local_pair_counts_overlapping():
    word = (b"A", b"A", b"A")

    result = get_local_pair_counts(word)

    assert result == Counter({
        (b"A", b"A"): 2,
    })


# test Merge Word
def test_merge_word():
    word = (
        b"A",
        b"B",
        b"C",
        b"A",
        b"B",
    )

    result = merge_word(
        word,
        (b"A", b"B"),
    )

    assert result == (
        b"AB",
        b"C",
        b"AB",
    )

def test_merge_word_non_overlapping():
    word = (
        b"A",
        b"A",
        b"A",
    )

    result = merge_word(
        word,
        (b"A", b"A"),
    )

    assert result == (
        b"AA",
        b"A",
    )

# test Build Pair Index
def test_build_pair_index():
    word_counts = Counter({
        (b"A", b"B", b"C"): 2,
        (b"A", b"B", b"B"): 3,
        (b"D", b"E"): 7,
    })

    (
        words,
        word_freqs,
        pair_counts,
        pair_to_words,
    ) = build_pair_index(word_counts)

    assert words == {
        0: (b"A", b"B", b"C"),
        1: (b"A", b"B", b"B"),
        2: (b"D", b"E"),
    }

    assert word_freqs == {
        0: 2,
        1: 3,
        2: 7,
    }

    assert pair_counts == Counter({
        (b"A", b"B"): 5,
        (b"B", b"C"): 2,
        (b"B", b"B"): 3,
        (b"D", b"E"): 7,
    })

    assert pair_to_words[(b"A", b"B")] == {0, 1}
    assert pair_to_words[(b"B", b"C")] == {0}
    assert pair_to_words[(b"B", b"B")] == {1}
    assert pair_to_words[(b"D", b"E")] == {2}

# test Repeated Pair Index

def test_build_pair_index_repeated_pair():
    word_counts = Counter({
        (b"A", b"A", b"A"): 5,
    })

    (
        words,
        word_freqs,
        pair_counts,
        pair_to_words,
    ) = build_pair_index(word_counts)

    assert pair_counts[(b"A", b"A")] == 10

    # Word occurs once in inverted index,
    # even though AA occurs twice inside it.
    assert pair_to_words[(b"A", b"A")] == {0}

# test Max Pair
def test_max_pair():
    pair_counts = Counter({
        (b"A", b"B"): 5,
        (b"C", b"D"): 7,
        (b"E", b"F"): 2,
    })

    assert max_pair(pair_counts) == (
        b"C",
        b"D",
    )

def test_max_pair():
    pair_counts = Counter({
        (b"A", b"B"): 5,
        (b"C", b"D"): 7,
        (b"E", b"F"): 2,
    })

    assert max_pair(pair_counts) == (
        b"C",
        b"D",
    )

# test Incremental Merge
def test_apply_incremental_merge():
    word_counts = Counter({
        (b"A", b"B", b"C"): 2,
        (b"A", b"B", b"B"): 3,
        (b"D", b"E"): 7,
    })

    (
        words,
        word_freqs,
        pair_counts,
        pair_to_words,
    ) = build_pair_index(word_counts)

    winner = max_pair(pair_counts)

    assert winner == (b"D", b"E")

    apply_incremental_merge(
        winner=winner,
        words=words,
        word_freqs=word_freqs,
        pair_counts=pair_counts,
        pair_to_words=pair_to_words,
    )

    assert words == {
        0: (b"A", b"B", b"C"),
        1: (b"A", b"B", b"B"),
        2: (b"DE",),
    }

    assert pair_counts == Counter({
        (b"A", b"B"): 5,
        (b"B", b"B"): 3,
        (b"B", b"C"): 2,
    })

    assert (b"D", b"E") not in pair_counts
    assert (b"D", b"E") not in pair_to_words


def test_two_incremental_merges():
    word_counts = Counter({
        (b"A", b"B", b"C"): 2,
        (b"A", b"B", b"B"): 3,
        (b"D", b"E"): 7,
    })

    (
        words,
        word_freqs,
        pair_counts,
        pair_to_words,
    ) = build_pair_index(word_counts)

    # Round 1: DE
    winner = max_pair(pair_counts)

    apply_incremental_merge(
        winner=winner,
        words=words,
        word_freqs=word_freqs,
        pair_counts=pair_counts,
        pair_to_words=pair_to_words,
    )

    # Round 2: AB
    winner = max_pair(pair_counts)

    assert winner == (b"A", b"B")

    apply_incremental_merge(
        winner=winner,
        words=words,
        word_freqs=word_freqs,
        pair_counts=pair_counts,
        pair_to_words=pair_to_words,
    )

    assert words == {
        0: (b"AB", b"C"),
        1: (b"AB", b"B"),
        2: (b"DE",),
    }

    assert pair_counts == Counter({
        (b"AB", b"B"): 3,
        (b"AB", b"C"): 2,
    })

    assert pair_to_words[(b"AB", b"B")] == {1}
    assert pair_to_words[(b"AB", b"C")] == {0}

# test: Differential test against Naive BPE
def write_corpus(tmp_path, text: str):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)

def test_optimized_matches_naive(tmp_path):
    input_path = write_corpus(
        tmp_path,
        """
        low low low lower
        newest newest
        widest widest widest
        """,
    )

    naive_vocab, naive_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=275,
        special_tokens=[],
    )

    optimized_vocab, optimized_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=275,
        special_tokens=[],
        num_workers=1,
    )

    assert optimized_merges == naive_merges
    assert optimized_vocab == naive_vocab

def test_optimized_matches_naive_overlapping(tmp_path):
    input_path = write_corpus(
        tmp_path,
        """
        aaaaa
        aaaa
        aaa
        aaaaa
        """,
    )

    naive_vocab, naive_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=265,
        special_tokens=[],
    )

    optimized_vocab, optimized_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=265,
        special_tokens=[],
        num_workers=1,
    )

    assert optimized_merges == naive_merges
    assert optimized_vocab == naive_vocab

def test_optimized_matches_naive_frequency_weighting(
    tmp_path,
):
    input_path = write_corpus(
        tmp_path,
        """
        abc abc abc abc abc
        def def
        xyz
        """,
    )

    naive_vocab, naive_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=270,
        special_tokens=[],
    )

    optimized_vocab, optimized_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=270,
        special_tokens=[],
        num_workers=1,
    )

    assert optimized_merges == naive_merges
    assert optimized_vocab == naive_vocab

# test: Special Tokens
def test_optimized_matches_naive_with_special_token(
    tmp_path,
):
    special_token = "<|endoftext|>"

    text = (
        "hello hello world"
        f"{special_token}"
        "hello newest newest"
        f"{special_token}"
        "world world world"
    )

    input_path = write_corpus(
        tmp_path,
        text,
    )

    naive_vocab, naive_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=275,
        special_tokens=[special_token],
    )

    optimized_vocab, optimized_merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=275,
        special_tokens=[special_token],
        num_workers=2,
    )

    assert optimized_merges == naive_merges
    assert optimized_vocab == naive_vocab

# test: Vocab Size
def test_optimized_vocab_size_too_small(tmp_path):
    input_path = write_corpus(
        tmp_path,
        "hello world",
    )

    with pytest.raises(ValueError):
        train_optimized_bpe(
            input_path=input_path,
            vocab_size=255,
            special_tokens=[],
            num_workers=1,
        )

def test_optimized_vocab_size_too_small_with_special_token(
    tmp_path,
):
    input_path = write_corpus(
        tmp_path,
        "hello<|endoftext|>world",
    )

    with pytest.raises(ValueError):
        train_optimized_bpe(
            input_path=input_path,
            vocab_size=256,
            special_tokens=["<|endoftext|>"],
            num_workers=1,
        )