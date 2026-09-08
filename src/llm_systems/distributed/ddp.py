import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()
        self.pending = []

        # Make every rank start with rank 0's parameters
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )

        # Launch gradient communication when each gradient
        # becomes available during backward
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(
                    self._on_gradient_ready
                )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _on_gradient_ready(
        self,
        parameter,
    ):
        work = dist.all_reduce(
            parameter.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        self.pending.append(
            (parameter, work)
        )

    def finish_gradient_synchronization(self):
        for parameter, work in self.pending:
            work.wait()

            parameter.grad /= self.world_size

        self.pending.clear()