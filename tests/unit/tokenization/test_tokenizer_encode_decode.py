from llm_systems.tokenization.tokenizer import Tokenizer


def test_encode_simple_bpe():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"ab",
        4: b"abc",
    }

    merges = [
        (b"a", b"b"),
        (b"ab", b"c"),
    ]

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
    )

    result = tokenizer.encode("abc")

    assert result == [4]


def test_encode_partial_merge():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"ab",
    }

    merges = [
        (b"a", b"b"),
    ]

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
    )

    result = tokenizer.encode("abc")

    assert result == [3, 2]


def test_decode():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"ab",
        4: b"abc",
    }

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=[],
    )

    result = tokenizer.decode([3, 2])

    assert result == "abc"


def test_encode_decode_round_trip():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"ab",
        4: b"abc",
    }

    merges = [
        (b"a", b"b"),
        (b"ab", b"c"),
    ]

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
    )

    text = "abc"

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert ids == [4]
    assert decoded == text

def test_encode_special_token():
    special_token = "<|endoftext|>"

    vocab = {
        0: b"a",
        1: b"b",
        2: b"ab",
        3: special_token.encode("utf-8"),
    }

    merges = [
        (b"a", b"b"),
    ]

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=[special_token],
    )

    text = f"ab{special_token}ab"

    result = tokenizer.encode(text)

    assert result == [2, 3, 2]

def test_decode_invalid_utf8_uses_replacement_character():
    vocab = {
        0: b"\xff",
    }

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=[],
    )

    result = tokenizer.decode([0])

    assert result == "\ufffd"