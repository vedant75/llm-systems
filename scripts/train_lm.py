import numpy as np
import torch
from pathlib import Path

from llm_systems.nn.transformer_lm import TransformerLM
from llm_systems.training.training_loop import train
from llm_systems.training.optimizer import AdamW
from llm_systems.training.checkpoint import load_checkpoint

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the custom Transformer language model."
    )

    # Data
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/tokenized/debug_sample.npy"),
    )

    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/tokenized/debug_sample.npy"),
    )

    # Model
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--d-ff",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
    )

    # Training
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--alpha-min",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cosine-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--eval-batches",
        type=int,
        default=2,
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "checkpoints/debug_training.pt"
        ),
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
    )

    # Device
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


def main():

    args = parse_args()

    # Device
    if args.device == 'auto':
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(f"Using device: {device}")

    # Dataset paths
    train_path = args.train_data

    val_path = args.val_data

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation data not found: {val_path}"
        )

    # Memory-mapped datasets
    train_data = np.load(
        train_path,
        mmap_mode="r",
    )

    val_data = np.load(
        val_path,
        mmap_mode="r",
    )

    print(
        f"Train: shape={train_data.shape}, "
        f"dtype={train_data.dtype}, "
        f"type={type(train_data)}"
    )

    print(
        f"Val: shape={val_data.shape}, "
        f"dtype={val_data.dtype}, "
        f"type={type(val_data)}"
    )

    # Construct model
    model_dtype = torch.float32
    
    model = TransformerLM(
        vocab_size= args.vocab_size,
        context_length= args.context_length,
        num_layers= args.num_layers,
        d_model= args.d_model,
        num_heads= args.num_heads,
        d_ff= args.d_ff,
        theta= args.rope_theta,
        device= device,
        dtype= model_dtype,
    )

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{num_parameters:,}"
    )

    # Construct optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
    )

    # Checkpoint configuration
    checkpoint_path = args.checkpoint_path

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_step = 0

    if args.resume_checkpoint is not None:

        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{args.resume_checkpoint}"
            )

        start_step = load_checkpoint(
            src=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
        )

        print(
            f"Resuming training from "
            f"step {start_step}"
        )

    else:
        print("Starting training from scratch")


    # Train
    print(
        f"Training from step "
        f"{start_step} to {args.num_steps}"
    )

    train(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        num_steps=args.num_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        warmup_steps=args.warmup_steps,
        cosine_steps=args.cosine_steps,
        eval_interval=args.eval_interval,
        start_step=start_step,
        checkpoint_path=str(args.checkpoint_path),
        checkpoint_interval=args.checkpoint_interval,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()