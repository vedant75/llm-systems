from __future__ import annotations

import torch
import torch.nn as nn

from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .attention import scaled_dot_product_attention

from jaxtyping import Float
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None
    )-> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads, "
                f"got d_model={d_model}, "
                f"num_heads={num_heads}."
            )

        head_dim = d_model // num_heads

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE, "
                f"got head_dim={head_dim}."
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.head_dim = head_dim

        self.Wq = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype
        )

        self.Wk = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype
        )

        self.Wv = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype
        )

        self.Wo = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype
        )

        self.rope = RotaryPositionalEmbedding(
            theta,
            head_dim,
            max_seq_len,
            device=device
        )

    def forward(
        self,
        x: Float[torch.Tensor, '... d_model'],
        token_pos
    ) -> Float[torch.Tensor, '... d_model']:

        seq = x.shape[-2] # get seq len

        # create q,k,v 
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # re-arrange the shapes of the q,k,v to pass it to attention
        q = rearrange(
            q,
            '... seq (head dim) -> ... head seq dim',
            head=self.num_heads
        )

        k = rearrange(
            k,
            '... seq (head dim) -> ... head seq dim',
            head=self.num_heads
        )

        v = rearrange(
            v,
            '... seq (head dim) -> ... head seq dim',
            head=self.num_heads
        )

        # apply rope: on Q, and K
        # Add singleton head axis so the same positional
        # rotations broadcast across every attention head.

        rope_position = token_pos.unsqueeze(-2)

        q = self.rope(
            q,
            rope_position
        )

        k = self.rope(
            k,
            rope_position
        )

        # create a mask
        mask = torch.ones(
            seq,
            seq,
            device=x.device,
            dtype=torch.bool
        ).tril()

        # apply attention
        attention_outputs = scaled_dot_product_attention(
            Q=q,
            K=k,
            V=v,
            mask=mask
        )

        # re-arrange the weights
        attention_outputs = rearrange(
            attention_outputs,
            '... head seq dim -> ... seq (head dim)'
        )

        return self.Wo(attention_outputs)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )