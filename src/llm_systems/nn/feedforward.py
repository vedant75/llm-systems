from __future__ import annotations

import torch
import torch.nn as nn

from jaxtyping import Float

from .linear import Linear


def silu(
        x: Float[torch.Tensor, "..."]
) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        if d_ff is None:
            d_ff = 64 * round(
                ((8.0 / 3.0) * d_model) / 64
            )

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype
        )
        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype
        )

        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype
        )


    def forward(
            self,
            x: Float[torch.Tensor, '... d_model']
    ) -> Float[torch.Tensor, '... d_model']:

        gate = silu(self.w1(x))
        value = self.w3(x)

        gated = gate * value

        return self.w2(gated)

    def extra_repr(self):
        return (
            f'd_model={self.d_model}, '
            f'd_ff={self.d_ff}'
        )