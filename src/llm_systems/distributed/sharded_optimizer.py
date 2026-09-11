from typing import Any

import torch
import torch.distributed as dist
from torch.optim.optimizer import Kwargs


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs,
    ) -> None:

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls

        self._next_param_index = 0

        # Parameter object -> rank responsible for updating it.
        self._param_to_rank = {}

        # Optimizer.__init__ internally calls
        # self.add_param_group().
        self._initializing = True

        super().__init__(
            params,
            defaults=kwargs,
        )

        self._initializing = False

        local_param_groups = []

        for param_group in self.param_groups:
            local_params = []

            for parameter in param_group["params"]:
                owner_rank = (
                    self._next_param_index
                    % self.world_size
                )

                self._param_to_rank[
                    id(parameter)
                ] = owner_rank

                if owner_rank == self.rank:
                    local_params.append(
                        parameter
                    )

                self._next_param_index += 1

            local_group = {
                key: value
                for key, value in param_group.items()
                if key != "params"
            }

            local_group["params"] = local_params

            local_param_groups.append(
                local_group
            )

        self.local_optimizer = optimizer_cls(
            local_param_groups,
            **kwargs,
        )

    def step(
        self,
        closure=None,
        **kwargs,
    ):
        loss = self.local_optimizer.step(
            closure=closure,
            **kwargs,
        )

        with torch.no_grad():
            for param_group in self.param_groups:
                for parameter in param_group['params']:
                    owner_rank = self._param_to_rank[
                        id(parameter)
                    ]

                    dist.broadcast(
                        parameter,
                        src=owner_rank,
                    )

        return loss

    def add_param_group(
        self,
        param_group: dict[str, Any],
    ) -> None:

        # Optimizer.__init__ calls add_param_group()
        # before self.local_optimizer exists.
        if self._initializing:
            super().add_param_group(
                param_group
            )
            return

        # Register the COMPLETE parameter group
        # in the ShardedOptimizer wrapper.
        super().add_param_group(
            param_group
        )

        full_group = self.param_groups[-1]

        # Determine which newly-added parameters
        # belong to this rank.
        local_params = []

        for parameter in full_group["params"]:

            owner_rank = (
                self._next_param_index
                % self.world_size
            )

            self._param_to_rank[
                id(parameter)
            ] = owner_rank

            if owner_rank == self.rank:
                local_params.append(
                    parameter
                )

            self._next_param_index += 1

        # Preserve lr, weight_decay, etc.,
        # but replace the full parameter list
        # with this rank's shard.
        local_group = {
            key: value
            for key, value in full_group.items()
            if key != "params"
        }

        local_group["params"] = local_params

        # Only the local optimizer owns optimizer
        # state for this rank's assigned parameters.
        self.local_optimizer.add_param_group(
            local_group
        )