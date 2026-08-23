from pathlib import Path

import numpy as np

from llm_systems.tokenization.tokenizer import Tokenizer


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: Tokenizer,
) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        token_iterator = tokenizer.encode_iterable(f)

        tokens = np.fromiter(
            token_iterator,
            dtype=np.uint16,
        )

    np.save(
        output_path,
        tokens,
    )

    print(f"Saved {len(tokens):,} tokens")
    print(f"Shape: {tokens.shape}")
    print(f"Dtype: {tokens.dtype}")
    print(f"Output: {output_path}")


# Small sample test
def main():
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    merges = []

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=None,
    )

    preprocess_file(
        input_path="data/debug_sample.txt",
        output_path="data/tokenized/debug_sample.npy",
        tokenizer=tokenizer,
    )

    data = np.load(
        "data/tokenized/debug_sample.npy",
        mmap_mode="r",
    )

    print(type(data))
    print(data.shape)
    print(data.dtype)
    print(data[:20])

    decoded = tokenizer.decode(
        data.tolist()
    )

    print(decoded)

if __name__ == "__main__":
    main()