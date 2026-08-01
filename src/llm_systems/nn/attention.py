from __future__ import annotations

import torch
import torch.nn as nn

import math
from einops import einsum

def softmax(
        x: torch.Tensor,
        dim: int 
) -> torch.Tensor:
    max_ = torch.max(
        x, 
        dim=dim, 
        keepdim=True
    ).values

    shifted_x = x - max_

    exp_x = torch.exp(shifted_x)

    denominator = exp_x.sum(
        dim=dim,
        keepdim=True,
    )

    return exp_x / denominator

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask = None
):
    dk = Q.shape[-1]
    # Q.shape: [... q_len dk]
    # K.shape: [... k_len dk]
    scores = einsum(
        Q,
        K, 
        '... q_len dk, ... k_len dk -> ... q_len k_len'
    )

    scores = scores/math.sqrt(dk)

    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    attention_weights = softmax(scores, dim=-1)

    # V.shape: [... k_len dv]
    output = einsum(
        attention_weights,
        V,
        '... q_len k_len, ... k_len dv -> ... q_len dv'
    )

    return output