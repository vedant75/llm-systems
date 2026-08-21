from __future__ import annotations

import torch

from jaxtyping import Float, Int

def cross_entropy_loss(
    logits: Float[torch.Tensor, '... V'],
    targets: Int[torch.Tensor, '...']
) -> Float[torch.Tensor, '...']:

    # Extract max logits of same shape: [B, T, V]
    max_logits = torch.max(
        logits,
        dim=-1,
        keepdim=True
    ).values

    # shifted logits
    shifted_logits = logits - max_logits

    # logits normalizer: log(exp(shifted_logits).sum(dim=-1))
    # shape: [B, T]
    logits_normalizer = (
        torch.log(
                torch.exp(shifted_logits)
                .sum(dim=-1)
        )
    )

    # Targets: [B, T]
    # for each [B, T] we want logits[B, T, targets[B, T]] -> [B, T]
    # gather the [B, T] with dim=-1 (vocab dim)
    target_logits = shifted_logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # calculate loss
    loss = logits_normalizer - target_logits

    return loss.mean()
