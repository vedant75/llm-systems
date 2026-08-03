from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(
                f"d_k must be even, received {d_k}."
            )

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Pair frequencies:
        # theta^(-0/d_k),
        # theta^(-2/d_k),
        # theta^(-4/d_k),
        inverse_frequencies = theta ** (
            -torch.arange(
                0,
                d_k,
                2,
                device=device,
                dtype=torch.float32,
            )
            / d_k
        )

        # Token positions:
        # [0, 1, 2, ..., max_seq_len - 1]
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        # [max_seq_len, d_k / 2]
        angles = torch.outer(
            positions,
            inverse_frequencies,
        )

        self.register_buffer(
            "cos",
            torch.cos(angles),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[
            torch.Tensor,
            "... seq_len"
        ],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:

        input_dtype = x.dtype

        # [..., seq_len, d_k] -> [..., seq_len, d_k / 2, 2]
        x_pairs = x.float().reshape(
            *x.shape[:-1],
            self.d_k // 2,
            2,
        )

        # First and second coordinate of every pair.
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        # [..., seq_len, d_k / 2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        rotated_x1 = (
            x1 * cos
            - x2 * sin
        )

        rotated_x2 = (
            x1 * sin
            + x2 * cos
        )

        rotated = torch.stack(
            [rotated_x1, rotated_x2],
            dim=-1,
        )

        # [..., seq_len, d_k / 2, 2] -> [..., seq_len, d_k]
        output = rotated.flatten(-2)

        return output.to(input_dtype)