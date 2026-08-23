import torch
import numpy as np

from .get_data import get_data
from .loss import cross_entropy_loss
from .checkpoint import save_checkpoint

from .training_helper import (
    learning_rate_schedule,
    gradient_clipping
)


def evaluate(
    model: torch.nn.Module,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int
) -> float:

    was_training = model.training

    model.eval()

    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_data(
                data=val_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )

            logits = model(inputs)

            loss = cross_entropy_loss(
                logits,
                targets
            )

            losses.append(loss.item())

    if was_training:
        model.train()

    return sum(losses) / len(losses)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    num_steps: int,
    eval_batches: int,
    batch_size: int,
    context_length: int,
    device: str,
    alpha_max: float,
    alpha_min: float,
    warmup_steps: int,
    cosine_steps: int,
    eval_interval: int,
    start_step: int = 0,
    checkpoint_path: str | None = None,
    checkpoint_interval: int | None = None,
    max_grad_norm: float = 1.0,
) -> None:

    model.train()

    for step in range(start_step, num_steps):

        # validation
        if step % eval_interval == 0:
            val_loss = evaluate(
                model = model,
                val_data = val_data,
                batch_size= batch_size,
                context_length = context_length,
                device = device,
                num_batches= eval_batches,
            )

            print(
                f"step={step} "
                f"val_loss={val_loss:.4f} "
            )


        # Current learning rate
        current_lr = learning_rate_schedule(
            t=step,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            t_w=warmup_steps,
            t_c=cosine_steps,
        )

        # Set current optimizer LR
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # Sample batch
        inputs, targets = get_data(
            data=train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Forward
        logits = model(inputs)

        # Loss
        loss = cross_entropy_loss(
            logits,
            targets,
        )

        # Backward
        loss.backward()

        # Clip gradients
        gradient_clipping(
            params=model.parameters(),
            max_norm=max_grad_norm,
        )

        # Update parameters
        optimizer.step()

        next_step = step + 1

        if step % 100 == 0:
            print(
                f"step={step} "
                f"loss={loss.item():.4f} "
                f"lr={current_lr:.6e}"
            )
        
        # save state
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and next_step % checkpoint_interval == 0
        ):
            save_checkpoint(
                model = model,
                optimizer= optimizer,
                iteration= next_step,
                out= checkpoint_path
            )