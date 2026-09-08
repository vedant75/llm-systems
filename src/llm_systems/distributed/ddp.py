import torch
import torch.distributed as dist
import torch.nn as nn


class MinimalDDP(nn.Module):
    """
    Naive DDP implementation.

    Waits for backward to finish, then synchronizes each
    parameter gradient individually.
    """

    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()

        # All ranks must start from rank 0's parameters.
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for parameter in self.module.parameters():

            # Some parameters may be frozen or unused.
            if parameter.grad is None:
                continue

            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
            )

            parameter.grad /= self.world_size


class FlatDDP(nn.Module):
    """
    DDP implementation that batches all gradients into
    one flattened tensor and performs one all-reduce.
    """

    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()

        # All ranks must start from rank 0's parameters.
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )

    def forward(
        self,
        *args,
        **kwargs,
    ):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):

        gradients = [
            parameter.grad
            for parameter in self.module.parameters()
            if parameter.grad is not None
        ]

        if not gradients:
            return

        # Pack all differently-shaped gradients into
        # one contiguous flat tensor.
        flat_grads = torch._utils._flatten_dense_tensors(
            gradients
        )

        # Only ONE collective call.
        dist.all_reduce(
            flat_grads,
            op=dist.ReduceOp.SUM,
        )

        flat_grads /= self.world_size

        # Recover tensors with the original gradient shapes.
        synced_gradients = torch._utils._unflatten_dense_tensors(
            flat_grads,
            gradients,
        )

        # Copy synchronized values back into param.grad.
        for original_grad, synced_grad in zip(
            gradients,
            synced_gradients,
        ):
            original_grad.copy_(synced_grad)


class OverlapDDP(nn.Module):
    """
    DDP implementation that asynchronously communicates
    each parameter gradient as soon as it becomes ready
    during backward.
    """

    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        super().__init__()

        self.module = module
        self.world_size = dist.get_world_size()

        # Outstanding asynchronous communication operations.
        self.pending = []

        # All ranks must start from rank 0's parameters.
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(
                    parameter,
                    src=0,
                )

        # Register a hook for every trainable parameter.
        #
        # When its gradient has been accumulated during
        # backward, _on_gradient_ready() is called automatically.
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
        # At this point parameter.grad contains
        # this rank's local gradient.
        if parameter.grad is None:
            return

        # Launch communication but do NOT wait here.
        work = dist.all_reduce(
            parameter.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        # Remember which gradient belongs to which
        # outstanding communication operation.
        self.pending.append(
            (parameter, work)
        )

    def finish_gradient_synchronization(self):
        for parameter, work in self.pending:

            # Ensure this gradient's all-reduce has finished.
            work.wait()

            # SUM -> average gradient.
            parameter.grad /= self.world_size

        self.pending.clear()