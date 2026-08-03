from __future__ import annotations

import torch
import torch.nn as nn

from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
            self,
            x: Float[torch.Tensor, '... d_model']
    ) -> Float[torch.Tensor, '... d_model']:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        squared_x = x**2
        mean_x = torch.mean(squared_x, dim=-1, keepdim=True)
        denom = torch.rsqrt(mean_x + self.eps)

        output = x * denom
        result = self.weight * output

        return result.to(in_dtype)

    def extra_repr(self):
        return (
            f'd_model: {self.d_model}'
        )