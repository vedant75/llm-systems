import torch

from llm_systems.training.optimizer import AdamW

def test_adamw_single_step_updates_parameter():
    param = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float32)
    )

    optimizer = AdamW(
        [param],
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    param.grad = torch.tensor([1.0])

    old_value = param.detach().clone()

    optimizer.step()

    assert param.item() < old_value.item()

def test_adamw_skips_parameter_without_gradient():
    param = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float32)
    )

    optimizer = AdamW(
        [param],
        lr=1e-2,
    )

    old_value = param.detach().clone()

    # param.grad is None
    optimizer.step()

    torch.testing.assert_close(
        param.detach(),
        old_value,
    )

def test_adamw_initializes_state():
    param = torch.nn.Parameter(
        torch.tensor([1.0, 2.0])
    )

    optimizer = AdamW(
        [param],
        lr=1e-3,
    )

    param.grad = torch.tensor([0.5, -0.5])

    optimizer.step()

    state = optimizer.state[param]

    assert state["t"] == 1

    assert state["m"].shape == param.shape
    assert state["v"].shape == param.shape

    assert torch.isfinite(state["m"]).all()
    assert torch.isfinite(state["v"]).all()

def test_adamw_timestep_increments():
    param = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    optimizer = AdamW(
        [param],
        lr=1e-3,
    )

    for step in range(1, 4):
        param.grad = torch.tensor([1.0])

        optimizer.step()

        assert optimizer.state[param]["t"] == step

def test_adamw_second_moment_uses_squared_gradient():
    param = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    beta2 = 0.999

    optimizer = AdamW(
        [param],
        lr=1e-3,
        betas=(0.9, beta2),
        weight_decay=0.0,
    )

    param.grad = torch.tensor([2.0])

    optimizer.step()

    state = optimizer.state[param]

    expected_v = (
        (1 - beta2)
        * torch.tensor([4.0])
    )

    torch.testing.assert_close(
        state["v"],
        expected_v,
    )

def test_adamw_first_moment_update():
    param = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    beta1 = 0.9

    optimizer = AdamW(
        [param],
        lr=1e-3,
        betas=(beta1, 0.999),
        weight_decay=0.0,
    )

    param.grad = torch.tensor([2.0])

    optimizer.step()

    expected_m = (
        (1 - beta1)
        * torch.tensor([2.0])
    )

    torch.testing.assert_close(
        optimizer.state[param]["m"],
        expected_m,
    )

def test_adamw_weight_decay():
    param = torch.nn.Parameter(
        torch.tensor([2.0])
    )

    lr = 0.1
    weight_decay = 0.2

    optimizer = AdamW(
        [param],
        lr=lr,
        weight_decay=weight_decay,
    )

    param.grad = torch.tensor([0.0])

    optimizer.step()

    expected = torch.tensor([
        2.0
        - lr * weight_decay * 2.0
    ])

    torch.testing.assert_close(
        param.detach(),
        expected,
    )

def test_adamw_matches_torch():
    torch.manual_seed(0)

    initial = torch.randn(
        3,
        4,
        dtype=torch.float32,
    )

    param_custom = torch.nn.Parameter(
        initial.clone()
    )

    param_torch = torch.nn.Parameter(
        initial.clone()
    )

    custom_optimizer = AdamW(
        [param_custom],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    torch_optimizer = torch.optim.AdamW(
        [param_torch],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    for _ in range(5):
        grad = torch.randn_like(
            param_custom
        )

        param_custom.grad = grad.clone()
        param_torch.grad = grad.clone()

        custom_optimizer.step()
        torch_optimizer.step()

    torch.testing.assert_close(
        param_custom.detach(),
        param_torch.detach(),
        rtol=1e-5,
        atol=1e-6,
    )

import math


def test_adamw_one_step_hand_calculated():
    param = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01

    optimizer = AdamW(
        [param],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    grad = torch.tensor([2.0])
    param.grad = grad.clone()

    optimizer.step()

    # Initial state:
    # m0 = 0
    # v0 = 0
    # t = 1

    m = (
        beta1 * 0
        + (1 - beta1) * 2.0
    )

    v = (
        beta2 * 0
        + (1 - beta2) * (2.0 ** 2)
    )

    alpha_t = (
        lr
        * math.sqrt(1 - beta2**1)
        / (1 - beta1**1)
    )

    after_decay = (
        1.0
        - lr * weight_decay * 1.0
    )

    expected = (
        after_decay
        - alpha_t
        * m
        / (
            math.sqrt(v)
            + eps
        )
    )

    torch.testing.assert_close(
        param.detach(),
        torch.tensor([expected]),
    )