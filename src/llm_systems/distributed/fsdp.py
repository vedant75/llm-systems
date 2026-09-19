from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from llm_systems.nn import Embedding, Linear


@dataclass
class ShardInfo:
    module: nn.Module
    parameter: nn.Parameter

    original_shape: torch.Size
    original_numel: int

    shard_numel: int
    padded_numel: int

    local_shard: torch.Tensor


class FSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.compute_dtype = compute_dtype

        self._shard_infos: list[ShardInfo] = []

        for submodule in self.module.modules():

            if not isinstance(
                submodule,
                (Linear, Embedding),
            ):
                continue

            self._shard_parameter(
                submodule,
                submodule.weight,
            )
        
        for info in self._shard_infos:
            self._register_forward_hooks(
                info
            )
            self._register_backward_hooks(info)

    def _shard_parameter(
        self,
        module: nn.Module,
        parameter: nn.Parameter,
    ) -> None:

        original_shape = parameter.shape
        original_numel = parameter.numel()

        shard_numel = (
            original_numel
            + self.world_size
            - 1
        ) // self.world_size

        padded_numel = (
            shard_numel
            * self.world_size
        )

        with torch.no_grad():

            flat = parameter.detach().reshape(-1)

            if padded_numel > original_numel:
                padded = torch.zeros(
                    padded_numel,
                    dtype=flat.dtype,
                    device=flat.device,
                )

                padded[:original_numel].copy_(
                    flat
                )

            else:
                padded = flat

            start = (
                self.rank
                * shard_numel
            )

            end = start + shard_numel

            local_shard = (
                padded[start:end]
                .clone()
                .to(torch.float32)
            )

            parameter.data = local_shard

        self._shard_infos.append(
            ShardInfo(
                module=module,
                parameter=parameter,
                original_shape=original_shape,
                original_numel=original_numel,
                shard_numel=shard_numel,
                padded_numel=padded_numel,
                local_shard=parameter.data,
            )
        )

    def forward(
        self,
        *args,
        **kwargs,
    ):
        return self.module(
            *args,
            **kwargs,
        )

    def _register_forward_hooks(
        self,
        info: ShardInfo,
    ) -> None:

        def pre_forward_hook(
            module,
            inputs,
        ):
            full_weight = self._gather_full_parameter(
                info
            )

            info.parameter.data = full_weight
        
        def post_forward_hook(
            module,
            inputs,
            output
        ):
            info.parameter.data = (
                info.local_shard
            )
        
        info.module.register_forward_pre_hook(
            pre_forward_hook
        )

        info.module.register_forward_hook(
            post_forward_hook
        )

    
    def _gather_full_parameter(
        self,
        info: ShardInfo,
    ) -> torch.Tensor:
        
        gathered = [
            torch.empty_like(
                info.local_shard
            )
            for _ in range(self.world_size)
        ]

        dist.all_gather(
            gathered,
            info.local_shard,
        )

        full_flat = torch.cat(
            gathered,
            dim=0  
        )

        full_flat = full_flat[
            :info.original_numel
        ]

        full_weight = full_flat.view(
            info.original_shape
        )

        return full_weight

    def _register_backward_hooks(
        self,
        info: ShardInfo,
    ) -> None:

        def pre_backward_hook(
            module,
            grad_output,
        ):
            full_weight = (
                self._gather_full_parameter(info)
            )

            info.parameter.data = full_weight

        def post_accumulate_grad_hook(
            parameter,
        ):
            parameter.data = (
                info.local_shard
            )

        info.module.register_full_backward_pre_hook(
            pre_backward_hook
        )

        info.parameter.register_post_accumulate_grad_hook(
            post_accumulate_grad_hook
        )
    
    def _reduce_scatter_gradient(
        self,
        info: ShardInfo,
        full_grad: torch.Tensor,
    ) -> torch.Tensor:

        flat_grad = full_grad.reshape(-1)

        padded_grad = torch.zeros(
            info.padded_numel,
            dtype=flat_grad.dtype,
            device=flat_grad.device,
        )

        padded_grad[
            :info.original_numel
        ].copy_(
            flat_grad
        )

        if dist.get_backend() == "gloo":

            # Correctness fallback:
            # all-reduce the complete gradient,
            # then keep only this rank's shard.
            dist.all_reduce(
                padded_grad,
                op=dist.ReduceOp.SUM,
            )

            padded_grad /= self.world_size

            start = (
                self.rank
                * info.shard_numel
            )

            end = start + info.shard_numel

            local_grad = (
                padded_grad[start:end]
                .clone()
            )

        else:

            local_grad = torch.empty(
                info.shard_numel,
                dtype=padded_grad.dtype,
                device=padded_grad.device,
            )

            dist.reduce_scatter_tensor(
                local_grad,
                padded_grad,
                op=dist.ReduceOp.SUM,
            )

            local_grad /= self.world_size

        return local_grad
    
    def finish_gradient_synchronization(
        self,
    ) -> None:

        sharded_parameter_ids = {
            id(info.parameter)
            for info in self._shard_infos
        }

        # Sharded parameters

        for info in self._shard_infos:

            parameter = info.parameter

            if parameter.grad is None:
                continue

            local_grad = (
                self._reduce_scatter_gradient(
                    info,
                    parameter.grad,
                )
            )

            # Master shards are FP32.
            local_grad = local_grad.to(
                info.local_shard.dtype
            )

            parameter.grad = local_grad

            # Be absolutely sure the parameter
            # itself is back in shard form.
            parameter.data = (
                info.local_shard
            )

        # Replicated parameters

        for parameter in self.module.parameters():

            if id(parameter) in sharded_parameter_ids:
                continue

            if parameter.grad is None:
                continue

            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
            )

            parameter.grad /= (
                self.world_size
            )