from time import perf_counter

from llm_systems.tokenization.bpe_tokenizer_optimized import (
    train_optimized_bpe,
)


def main():
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"

    start = perf_counter()

    vocab, merges = train_optimized_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        num_workers=4,
    )

    elapsed = perf_counter() - start

    print(f"Runtime: {elapsed:.3f} seconds")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")


if __name__ == "__main__":
    main()