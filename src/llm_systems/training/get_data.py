from __future__ import annotations

import numpy as np
import torch


def get_data(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:

    # We need T input tokens + 1 token
    # for the final next-token target.
    starts = np.random.randint(
        low=0,
        high=len(data) - context_length,
        size=batch_size,
    )

    inputs = np.stack(
        [
            data[
                start : start + context_length
            ]
            for start in starts
        ],
        axis=0,
    )

    targets = np.stack(
        [
            data[
                start + 1 :
                start + context_length + 1
            ]
            for start in starts
        ],
        axis=0,
    )

    inputs = torch.from_numpy(
        inputs
    ).to(
        device=device,
        dtype=torch.long,
    )

    targets = torch.from_numpy(
        targets
    ).to(
        device=device,
        dtype=torch.long,
    )

    return inputs, targets