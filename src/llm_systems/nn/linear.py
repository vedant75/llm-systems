from __future__ import annotations

import math

import torch
import torch.nn as nn

from jaxtyping import Float
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0/(self.in_features + self.out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a = -3*std,
            b = 3 * std
        )

    def forward(
            self,
            x:Float[torch.Tensor, '... d_in']
        ) -> Float[torch.Tensor, '... d_out']:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

    def extra_repr(self) -> str:
        return (
            f'in_features: {self.in_features}, '
            f'out_features: {self.out_features}'
        )