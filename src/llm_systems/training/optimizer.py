from collections.abc import Callable
from typing import Optional

import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ) -> None:

        if lr < 0:
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(
            params,
            defaults,
        )

    def step(
        self,
        closure: Optional[Callable] = None,
    ):

        loss = (
            None
            if closure is None
            else closure()
        )

        for group in self.param_groups:

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                # Initialize state

                if len(state) == 0:
                    state["t"] = 0

                    state["m"] = torch.zeros_like(
                        p.data
                    )

                    state["v"] = torch.zeros_like(
                        p.data
                    )

                # Increment timestep

                state["t"] += 1

                t = state["t"]

                m = state["m"]
                v = state["v"]

                # Adjusted learning rate

                alpha_t = (
                    lr
                    * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                )

                # Decoupled weight decay

                p.data -= (
                    lr
                    * weight_decay
                    * p.data
                )

                # First moment

                m.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # Second moment

                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # Adam update

                p.data -= (
                    alpha_t
                    * m / (torch.sqrt(v) + eps)
                )

        return loss