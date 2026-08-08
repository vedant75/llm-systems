from collections import Counter

from llm_systems.tokenization.bpe_reference_helper import (
    get_most_frequent_pair, 
    apply_merge
)

from llm_systems.tokenization.bpe_tokenizer import train_bpe


def to_symbols(text: str) -> tuple[bytes, ...]:
    return tuple(
        bytes([b])
        for b in text.encode("utf-8")
    )


def test_most_frequent_pair_uses_word_frequency():
    words = Counter({
        to_symbols("low"): 5,
        to_symbols("lower"): 2,
        to_symbols("widest"): 3,
        to_symbols("newest"): 6,
    })

    pair = get_most_frequent_pair(words)

    assert pair == (b"s", b"t")

def test_apply_merge():
    words = Counter({
        (
            b"n",
            b"e",
            b"w",
            b"e",
            b"s",
            b"t",
        ): 6
    })

    result = apply_merge(
        words,
        (b"s", b"t"),
    )

    expected = Counter({
        (
            b"n",
            b"e",
            b"w",
            b"e",
            b"st",
        ): 6
    })

    assert result == expected


def test_special_tokens_do_not_contribute_to_merges(tmp_path):
    corpus = tmp_path / "corpus.txt"

    corpus.write_text(
        "aa<|endoftext|>aa",
        encoding="utf-8",
    )

    vocab, merges = train_bpe(
        input_path=str(corpus),
        vocab_size=258,
        special_tokens=["<|endoftext|>"],
    )

    # 256 bytes + 1 special token + 1 merge
    assert len(vocab) == 258

    assert merges == [
        (b"a", b"a")
    ]

    assert vocab[256] == b"<|endoftext|>"
    assert vocab[257] == b"aa"