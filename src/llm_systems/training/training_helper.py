from __future__ import annotations
from collections.abc import Iterable

import math
import torch


def learning_rate_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    t_w: int,
    t_c: int,
) -> float:
    # warm-up
    if t < t_w:
        return (t/t_w)*alpha_max

    # cosine-annealing
    elif t_w <= t <= t_c:
        progress = (t - t_w)/(t_c - t_w)
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))

        return alpha_min + cosine_decay*(alpha_max - alpha_min)

    # post-annealing
    else:
        return alpha_min


# Gradient Clipping

def gradient_clipping(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
) -> None:

    eps = 1e-6

    total_squared = None

    params = list(params)

    for p in params:
        if p.grad is None:
            continue

        gradient_squared = (
            torch.sum(p.grad ** 2)
        )

        if total_squared is None:
            total_squared = gradient_squared
        else:
            total_squared += gradient_squared
        
    if total_squared is None:
        return

    norm = torch.sqrt(
        total_squared
    )

    if norm > max_norm:
        scale = (
            max_norm / (norm + eps)
        )

        for p in params:
            if p.grad is None:
                continue
            
            p.grad.mul_(scale)