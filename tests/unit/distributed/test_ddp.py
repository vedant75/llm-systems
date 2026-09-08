# tests/test_ddp.py

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from llm_systems.distributed.ddp import DDP


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)


def find_free_port():
    """
    Find an unused local port so that repeated pytest runs
    don't conflict with an old distributed process group.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
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


def assert_parameters_same_across_ranks(model):
    """
    Verify that every rank has exactly the same model parameters.
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

        for rank_parameter in gathered[1:]:
            assert torch.allclose(
                reference,
                rank_parameter,
                atol=1e-6,
                rtol=1e-5,
            )


def assert_gradients_same_across_ranks(model):
    """
    Verify that synchronized gradients are identical on every rank.
    """
    world_size = dist.get_world_size()

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        assert parameter.grad is not None

        gathered = [
            torch.empty_like(parameter.grad)
            for _ in range(world_size)
        ]

        dist.all_gather(
            gathered,
            parameter.grad,
        )

        reference = gathered[0]

        for rank_gradient in gathered[1:]:
            assert torch.allclose(
                reference,
                rank_gradient,
                atol=1e-6,
                rtol=1e-5,
            )


def _test_ddp_worker(
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
    # 1. Give every rank a DIFFERENT initial model
    # --------------------------------------------------

    torch.manual_seed(rank)

    baseline_model = ToyModel()

    # Make a separate model with the same LOCAL
    # initialization before DDP changes anything.
    ddp_base = ToyModel()
    ddp_base.load_state_dict(
        baseline_model.state_dict()
    )

    # Wrapping should broadcast rank 0 parameters.
    ddp_model = DDP(ddp_base)

    # --------------------------------------------------
    # 2. Verify initialization synchronization
    # --------------------------------------------------

    assert_parameters_same_across_ranks(
        ddp_model
    )

    # Rank 0 should still equal its original model.
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
    # 3. Create one GLOBAL dataset
    # --------------------------------------------------

    # Every process must create exactly the same dataset.
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

    baseline_optimizer = optim.SGD(
        baseline_model.parameters(),
        lr=0.1,
    )

    # --------------------------------------------------
    # 4. Train for multiple iterations
    # --------------------------------------------------

    for step in range(5):

        ddp_optimizer.zero_grad()
        baseline_optimizer.zero_grad()

        # ==============================================
        # Normal single-process baseline
        #
        # Only rank 0 needs this for our comparison.
        # It sees the COMPLETE batch.
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
        # DDP
        #
        # Each rank sees a DISJOINT shard.
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

        # Our hooks launched asynchronous all-reduces
        # during backward.
        #
        # Now wait for them and average the gradients.
        ddp_model.finish_gradient_synchronization()

        # --------------------------------------------------
        # 5. Verify gradients are identical across ranks
        # --------------------------------------------------

        assert_gradients_same_across_ranks(
            ddp_model
        )

        # --------------------------------------------------
        # 6. Baseline gradient should equal DDP gradient
        # --------------------------------------------------

        if rank == 0:
            for baseline_parameter, ddp_parameter in zip(
                baseline_model.parameters(),
                ddp_model.parameters(),
            ):
                assert torch.allclose(
                    baseline_parameter.grad,
                    ddp_parameter.grad,
                    atol=1e-6,
                    rtol=1e-5,
                )

        # --------------------------------------------------
        # 7. Apply optimizer updates
        # --------------------------------------------------

        ddp_optimizer.step()

        if rank == 0:
            baseline_optimizer.step()

        # --------------------------------------------------
        # 8. All DDP replicas should remain identical
        # --------------------------------------------------

        assert_parameters_same_across_ranks(
            ddp_model
        )

        # --------------------------------------------------
        # 9. Rank-0 DDP should match full-batch baseline
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
        # 10. Shuffle for the next step
        # --------------------------------------------------

        torch.manual_seed(42 + step)

        permutation = torch.randperm(
            all_x.size(0)
        )

        all_x = all_x[permutation]
        all_y = all_y[permutation]

    cleanup_process_group()


def test_ddp():
    world_size = 2

    port = find_free_port()

    mp.spawn(
        _test_ddp_worker,
        args=(
            world_size,
            port,
        ),
        nprocs=world_size,
        join=True,
    )