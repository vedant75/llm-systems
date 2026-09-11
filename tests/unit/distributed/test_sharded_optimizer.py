import os
import socket
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from llm_systems.distributed.sharded_optimizer import ShardedOptimizer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
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


def assert_models_equal(
    model_a: nn.Module,
    model_b: nn.Module,
):
    for parameter_a, parameter_b in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):
        assert torch.allclose(
            parameter_a,
            parameter_b,
            atol=1e-6,
            rtol=1e-5,
        )


def assert_model_equal_across_ranks(
    model: nn.Module,
):
    """
    Verify every rank has the same complete model.
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


def _test_basic_sharded_optimizer(
    rank: int,
    world_size: int,
    port: int,
):
    setup_process_group(
        rank,
        world_size,
        port,
    )

    # --------------------------------------------------
    # 1. Every rank starts with the SAME model.
    # --------------------------------------------------

    torch.manual_seed(0)

    baseline_model = ToyModel()
    sharded_model = deepcopy(baseline_model)

    # Same deterministic data on every rank.
    torch.manual_seed(1234)

    x = torch.randn(16, 4)
    y = torch.randn(16, 2)

    loss_fn = nn.MSELoss()

    # --------------------------------------------------
    # 2. Ordinary AdamW baseline.
    # --------------------------------------------------

    baseline_optimizer = optim.AdamW(
        baseline_model.parameters(),
        lr=1e-2,
        weight_decay=0.01,
    )

    # --------------------------------------------------
    # 3. Our sharded AdamW.
    # --------------------------------------------------

    sharded_optimizer = ShardedOptimizer(
        sharded_model.parameters(),
        optim.AdamW,
        lr=1e-2,
        weight_decay=0.01,
    )

    # --------------------------------------------------
    # 4. Verify parameter ownership.
    #
    # With round-robin ownership:
    #
    # P0 -> rank 0
    # P1 -> rank 1
    # P2 -> rank 0
    # P3 -> rank 1
    # --------------------------------------------------

    all_parameters = list(
        sharded_model.parameters()
    )

    expected_local_ids = {
        id(parameter)
        for index, parameter
        in enumerate(all_parameters)
        if index % world_size == rank
    }

    actual_local_ids = {
        id(parameter)
        for group
        in sharded_optimizer.local_optimizer.param_groups
        for parameter
        in group["params"]
    }

    assert (
        actual_local_ids
        == expected_local_ids
    )

    # --------------------------------------------------
    # 5. Train both implementations.
    # --------------------------------------------------

    for step in range(5):

        baseline_optimizer.zero_grad()
        sharded_optimizer.zero_grad()

        # ------------------------------
        # Ordinary AdamW
        # ------------------------------

        baseline_output = baseline_model(x)

        baseline_loss = loss_fn(
            baseline_output,
            y,
        )

        baseline_loss.backward()

        baseline_optimizer.step()

        # ------------------------------
        # Sharded AdamW
        # ------------------------------

        sharded_output = sharded_model(x)

        sharded_loss = loss_fn(
            sharded_output,
            y,
        )

        sharded_loss.backward()

        sharded_optimizer.step()

        # --------------------------------------------------
        # 6. Sharded result should reproduce normal AdamW.
        # --------------------------------------------------

        assert_models_equal(
            baseline_model,
            sharded_model,
        )

        # --------------------------------------------------
        # 7. Every rank should have the SAME full model.
        # --------------------------------------------------

        assert_model_equal_across_ranks(
            sharded_model
        )

        # --------------------------------------------------
        # 8. After the first optimizer step, AdamW state
        #    should exist ONLY for locally-owned parameters.
        # --------------------------------------------------

        state_parameter_ids = {
            id(parameter)
            for parameter
            in sharded_optimizer.local_optimizer.state.keys()
        }

        assert (
            state_parameter_ids
            == expected_local_ids
        )

    cleanup_process_group()


def test_sharded_optimizer():
    world_size = 2
    port = find_free_port()

    mp.spawn(
        _test_basic_sharded_optimizer,
        args=(
            world_size,
            port,
        ),
        nprocs=world_size,
        join=True,
    )


# ============================================================
# add_param_group test
# ============================================================


def _test_add_param_group(
    rank: int,
    world_size: int,
    port: int,
):
    setup_process_group(
        rank,
        world_size,
        port,
    )

    torch.manual_seed(0)

    baseline_model = ToyModel()
    sharded_model = deepcopy(
        baseline_model
    )

    torch.manual_seed(5678)

    x = torch.randn(16, 4)
    y = torch.randn(16, 2)

    loss_fn = nn.MSELoss()

    # --------------------------------------------------
    # Initially optimize ONLY linear1.
    # --------------------------------------------------

    baseline_optimizer = optim.AdamW(
        baseline_model.linear1.parameters(),
        lr=1e-2,
    )

    sharded_optimizer = ShardedOptimizer(
        sharded_model.linear1.parameters(),
        optim.AdamW,
        lr=1e-2,
    )

    # --------------------------------------------------
    # One step with only linear1.
    # --------------------------------------------------

    baseline_model.zero_grad()
    sharded_model.zero_grad()

    baseline_loss = loss_fn(
        baseline_model(x),
        y,
    )
    baseline_loss.backward()
    baseline_optimizer.step()

    sharded_loss = loss_fn(
        sharded_model(x),
        y,
    )
    sharded_loss.backward()
    sharded_optimizer.step()

    assert_models_equal(
        baseline_model,
        sharded_model,
    )

    # --------------------------------------------------
    # NOW add linear2 as a new parameter group.
    # --------------------------------------------------

    baseline_optimizer.add_param_group(
        {
            "params":
                baseline_model.linear2.parameters(),
            "lr": 5e-3,
        }
    )

    sharded_optimizer.add_param_group(
        {
            "params":
                sharded_model.linear2.parameters(),
            "lr": 5e-3,
        }
    )

    assert len(
        sharded_optimizer.param_groups
    ) == 2

    # --------------------------------------------------
    # Train after adding the new group.
    # --------------------------------------------------

    for _ in range(3):

        baseline_model.zero_grad()
        sharded_model.zero_grad()

        baseline_loss = loss_fn(
            baseline_model(x),
            y,
        )
        baseline_loss.backward()
        baseline_optimizer.step()

        sharded_loss = loss_fn(
            sharded_model(x),
            y,
        )
        sharded_loss.backward()
        sharded_optimizer.step()

        assert_models_equal(
            baseline_model,
            sharded_model,
        )

        assert_model_equal_across_ranks(
            sharded_model
        )

    cleanup_process_group()


def test_sharded_optimizer_add_param_group():
    world_size = 2
    port = find_free_port()

    mp.spawn(
        _test_add_param_group,
        args=(
            world_size,
            port,
        ),
        nprocs=world_size,
        join=True,
    )