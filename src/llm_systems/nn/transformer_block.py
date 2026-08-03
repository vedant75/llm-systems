from __future__ import annotations

import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .multiheadattention import MultiHeadAttention
from .feedforward import SwiGLU

from jaxtyping import Float

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.sa = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )

        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        x: Float[torch.Tensor, '... d_model'],
        token_pos
    ) -> Float[torch.Tensor, '... d_model']:
        x = x + self.sa(self.ln1(x), token_pos)
        x = x + self.ffn(self.ln1(x))
        return x