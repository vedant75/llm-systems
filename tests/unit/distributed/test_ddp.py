import os
import socket
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from llm_systems.distributed.ddp import (
    MinimalDDP,
    FlatDDP,
    OverlapDDP,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 10)

        # Useful edge case:
        # this parameter should never receive a gradient.
        self.no_grad_fixed_param = nn.Parameter(
            torch.tensor([2.0, 2.0]),
            requires_grad=False,
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)


def find_free_port():
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def setup_process_group(
    rank: int,
    world_size: int,
    port: int,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )


def cleanup_process_group():
    dist.destroy_process_group()


def assert_models_equal_across_ranks(
    model: nn.Module,
):
    """
    Check that every rank has exactly the same parameters.
    """
    world_size = dist.get_world_size()

    for parameter in model.parameters():

        gathered = [
            torch.empty_like(parameter)
            for _ in range(world_size)
        ]

        dist.all_gather(
            gathered,
            parameter.detach(),
        )

        reference = gathered[0]

        for other in gathered[1:]:
            assert torch.allclose(
                reference,
                other,
                atol=1e-6,
                rtol=1e-5,
            )


def assert_gradients_equal_across_ranks(
    model: nn.Module,
):
    """
    Check that every synchronized gradient is identical
    across all ranks.
    """
    world_size = dist.get_world_size()

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        gathered = [
            torch.empty_like(parameter.grad)
            for _ in range(world_size)
        ]

        dist.all_gather(
            gathered,
            parameter.grad,
        )

        reference = gathered[0]

        for other in gathered[1:]:
            assert torch.allclose(
                reference,
                other,
                atol=1e-6,
                rtol=1e-5,
            )


def _ddp_worker(
    rank: int,
    world_size: int,
    port: int,
    ddp_type: str,
):
    setup_process_group(
        rank,
        world_size,
        port,
    )

    # --------------------------------------------------
    # 1. Make each rank start from DIFFERENT weights.
    # --------------------------------------------------

    torch.manual_seed(rank)

    local_model = ToyModel()

    # Rank 0's model is our full-batch reference.
    if rank == 0:
        baseline_model = deepcopy(local_model)

    # --------------------------------------------------
    # 2. Wrap using the requested DDP implementation.
    # --------------------------------------------------

    if ddp_type == "minimal":
        ddp_model = MinimalDDP(local_model)

    elif ddp_type == "flat":
        ddp_model = FlatDDP(local_model)

    elif ddp_type == "overlap":
        ddp_model = OverlapDDP(local_model)

    else:
        raise ValueError(
            f"Unknown DDP type: {ddp_type}"
        )

    # DDP constructor should have broadcast rank 0's
    # parameters to every process.
    assert_models_equal_across_ranks(
        ddp_model
    )

    # Rank 0 should still exactly match its original model.
    if rank == 0:
        for baseline_parameter, ddp_parameter in zip(
            baseline_model.parameters(),
            ddp_model.parameters(),
        ):
            assert torch.allclose(
                baseline_parameter,
                ddp_parameter,
            )

    # --------------------------------------------------
    # 3. Create identical GLOBAL data on every rank.
    # --------------------------------------------------

    torch.manual_seed(1234)

    all_x = torch.randn(20, 10)
    all_y = torch.randn(20, 10)

    assert all_x.size(0) % world_size == 0

    local_batch_size = (
        all_x.size(0) // world_size
    )

    loss_fn = nn.MSELoss()

    ddp_optimizer = optim.SGD(
        ddp_model.parameters(),
        lr=0.1,
    )

    if rank == 0:
        baseline_optimizer = optim.SGD(
            baseline_model.parameters(),
            lr=0.1,
        )

    # --------------------------------------------------
    # 4. Run several training iterations.
    # --------------------------------------------------

    for step in range(5):

        ddp_optimizer.zero_grad()

        if rank == 0:
            baseline_optimizer.zero_grad()

        # ==============================================
        # Full-batch baseline
        # ==============================================

        if rank == 0:
            baseline_output = baseline_model(
                all_x
            )

            baseline_loss = loss_fn(
                baseline_output,
                all_y,
            )

            baseline_loss.backward()

        # ==============================================
        # Distributed path
        #
        # Each rank receives a disjoint half of the batch.
        # ==============================================

        start = rank * local_batch_size
        end = start + local_batch_size

        local_x = all_x[start:end]
        local_y = all_y[start:end]

        ddp_output = ddp_model(
            local_x
        )

        ddp_loss = loss_fn(
            ddp_output,
            local_y,
        )

        ddp_loss.backward()

        # This performs the different synchronization
        # strategy for Minimal / Flat / Overlap DDP.
        ddp_model.finish_gradient_synchronization()

        # --------------------------------------------------
        # 5. All ranks must now hold the same gradients.
        # --------------------------------------------------

        assert_gradients_equal_across_ranks(
            ddp_model
        )

        # --------------------------------------------------
        # 6. DDP gradient should equal full-batch gradient.
        # --------------------------------------------------

        if rank == 0:

            for baseline_parameter, ddp_parameter in zip(
                baseline_model.parameters(),
                ddp_model.parameters(),
            ):

                if baseline_parameter.grad is None:
                    assert ddp_parameter.grad is None
                    continue

                assert torch.allclose(
                    baseline_parameter.grad,
                    ddp_parameter.grad,
                    atol=1e-6,
                    rtol=1e-5,
                )

        # --------------------------------------------------
        # 7. Take optimizer steps.
        # --------------------------------------------------

        ddp_optimizer.step()

        if rank == 0:
            baseline_optimizer.step()

        # --------------------------------------------------
        # 8. All distributed replicas must stay identical.
        # --------------------------------------------------

        assert_models_equal_across_ranks(
            ddp_model
        )

        # --------------------------------------------------
        # 9. DDP should exactly reproduce full-batch training.
        # --------------------------------------------------

        if rank == 0:

            for baseline_parameter, ddp_parameter in zip(
                baseline_model.parameters(),
                ddp_model.parameters(),
            ):
                assert torch.allclose(
                    baseline_parameter,
                    ddp_parameter,
                    atol=1e-6,
                    rtol=1e-5,
                )

        # --------------------------------------------------
        # 10. Shuffle identically before next iteration.
        # --------------------------------------------------

        torch.manual_seed(42 + step)

        permutation = torch.randperm(
            all_x.size(0)
        )

        all_x = all_x[permutation]
        all_y = all_y[permutation]

    cleanup_process_group()


@pytest.mark.parametrize(
    "ddp_type",
    [
        "minimal",
        "flat",
        "overlap",
    ],
)
def test_ddp_correctness(
    ddp_type: str,
):
    world_size = 2
    port = find_free_port()

    mp.spawn(
        _ddp_worker,
        args=(
            world_size,
            port,
            ddp_type,
        ),
        nprocs=world_size,
        join=True,
    )