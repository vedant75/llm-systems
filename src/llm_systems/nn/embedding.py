from __future__ import annotations

import torch
import torch.nn as nn

from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError(
                'num_embeddings must be positive, '
                f'received: {num_embeddings}'
            )

        if embedding_dim <= 0:
            raise ValueError(
                'embedding_dim must be positive, '
                f'received: {embedding_dim}'
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1,
            a=-3,
            b=3
        )

    def forward(
            self,
            token_ids: Int[torch.Tensor, '...']
    ) -> Float[torch.Tensor, '... embedding_dim']:
        return (
            self.weight[token_ids]
        )

    def extra_repr(self):
        return (
            f'num_embeddings: {self.num_embeddings}, '
            f'embedding_dim: {self.embedding_dim}'
        )