import argparse
from pathlib import Path

import torch

from llm_systems.nn.transformer_lm import TransformerLM
from llm_systems.tokenization.tokenizer import Tokenizer
from llm_systems.generation.decoding import generate

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Transformer."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()

def build_debug_tokenizer() -> Tokenizer:
    vocab = {
        i: bytes([i])
        for i in range(256)
    }

    merges = []

    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=None,
    )


def main():

    args = parse_args()

    # Device
    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    print(f"Using device: {device}")

    # Checkpoint
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    # Tokenizer
    tokenizer = build_debug_tokenizer()

    # Model configuration
    # Must match training configuration.

    vocab_size = 256
    context_length = 16

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=128,
        theta=10000.0,
        device=device,
        dtype=torch.float32,
    )

    # Load model weights
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    iteration = checkpoint.get(
        "iteration",
        "unknown",
    )

    print(
        f"Loaded checkpoint from "
        f"iteration {iteration}"
    )

    # Generate
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=None,
    )

    print()
    print("Generated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()