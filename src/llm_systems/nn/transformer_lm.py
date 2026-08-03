from __future__ import annotations

import torch
import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from.transformer_block import TransformerBlock

from jaxtyping import Float, Int

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model

        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        idx: Int[torch.Tensor, '... seq']
    ) -> Float[torch.Tensor, '... d_model']:

        seq_len = idx.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"context length {self.context_length}."
            )
        
        x = self.token_embedding(idx) # B, T, D

        # 2. self.transformer block
        positions = torch.arange(
            seq_len,
            device=idx.device,
            dtype=torch.long
        )

        for layer in self.layers:
            x = layer(x, positions)

        x = self.ln_final(x) # B, T, D
        logits = self.lm_head(x) # B, T, vocab_size

        return logits

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"context_length={self.context_length}, "
            f"num_layers={self.num_layers}, "
            f"d_model={self.d_model}"
        )