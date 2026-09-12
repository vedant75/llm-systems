from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

# Change this import to wherever your own classes live.
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