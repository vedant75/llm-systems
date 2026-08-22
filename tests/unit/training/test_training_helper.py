import math
import torch

from llm_systems.training.training_helper import (
    learning_rate_schedule,
    gradient_clipping
)


def test_learning_rate_schedule():
    alpha_max = 1e-3
    alpha_min = 1e-4
    t_w = 100
    t_c = 1000

    # Start of warmup
    assert learning_rate_schedule(
        0,
        alpha_max,
        alpha_min,
        t_w,
        t_c,
    ) == 0.0

    # End of warmup / start of cosine
    assert math.isclose(
        learning_rate_schedule(
            t_w,
            alpha_max,
            alpha_min,
            t_w,
            t_c,
        ),
        alpha_max,
    )

    # Middle of cosine schedule
    midpoint = (t_w + t_c) // 2

    expected_midpoint = (
        alpha_min
        + 0.5 * (alpha_max - alpha_min)
    )

    assert math.isclose(
        learning_rate_schedule(
            midpoint,
            alpha_max,
            alpha_min,
            t_w,
            t_c,
        ),
        expected_midpoint,
        rel_tol=1e-6,
    )

    # End of cosine
    assert math.isclose(
        learning_rate_schedule(
            t_c,
            alpha_max,
            alpha_min,
            t_w,
            t_c,
        ),
        alpha_min,
    )

    # Post-annealing
    assert math.isclose(
        learning_rate_schedule(
            t_c + 100,
            alpha_max,
            alpha_min,
            t_w,
            t_c,
        ),
        alpha_min,
    )

# Gradient Clipping

def test_gradient_clipping():
    param = torch.nn.Parameter(
        torch.tensor([1.0, 2.0])
    )

    param.grad = torch.tensor(
        [3.0, 4.0]
    )

    # Original gradient norm:
    # sqrt(3^2 + 4^2) = 5

    gradient_clipping(
        [param],
        max_norm=1.0,
    )

    expected_scale = (
        1.0 / (5.0 + 1e-6)
    )

    expected_grad = torch.tensor(
        [3.0, 4.0]
    ) * expected_scale

    torch.testing.assert_close(
        param.grad,
        expected_grad,
    )

    # New norm should be at or just below 1.
    assert torch.linalg.vector_norm(
        param.grad
    ) <= 1.0

def test_gradient_clipping_does_not_clip_small_gradient():
    param = torch.nn.Parameter(
        torch.tensor([1.0, 2.0])
    )

    param.grad = torch.tensor(
        [0.3, 0.4]
    )

    # norm = 0.5
    original_grad = param.grad.clone()

    gradient_clipping(
        [param],
        max_norm=1.0,
    )

    torch.testing.assert_close(
        param.grad,
        original_grad,
    )

def test_gradient_clipping_uses_global_norm():
    p1 = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    p2 = torch.nn.Parameter(
        torch.tensor([2.0])
    )

    p1.grad = torch.tensor([3.0])
    p2.grad = torch.tensor([4.0])

    gradient_clipping(
        [p1, p2],
        max_norm=1.0,
    )

    expected_scale = (
        1.0 / (5.0 + 1e-6)
    )

    torch.testing.assert_close(
        p1.grad,
        torch.tensor([3.0])
        * expected_scale,
    )

    torch.testing.assert_close(
        p2.grad,
        torch.tensor([4.0])
        * expected_scale,
    )

    total_norm = torch.sqrt(
        torch.sum(p1.grad ** 2)
        + torch.sum(p2.grad ** 2)
    )

    assert total_norm <= 1.0

def test_gradient_clipping_skips_none_gradient():
    p1 = torch.nn.Parameter(
        torch.tensor([1.0])
    )

    p2 = torch.nn.Parameter(
        torch.tensor([2.0])
    )

    p1.grad = torch.tensor([3.0])

    # p2.grad remains None

    gradient_clipping(
        [p1, p2],
        max_norm=1.0,
    )

    assert p2.grad is None

    assert torch.linalg.vector_norm(
        p1.grad
    ) <= 1.0

def test_gradient_clipping_accepts_model_parameters():
    model = torch.nn.Linear(
        2,
        1,
        bias=False,
    )

    for param in model.parameters():
        param.grad = torch.full_like(
            param,
            10.0,
        )

    # model.parameters() is passed directly,
    # rather than manually converting it to a list.
    gradient_clipping(
        model.parameters(),
        max_norm=1.0,
    )

    total_squared = 0.0

    for param in model.parameters():
        total_squared += (
            param.grad ** 2
        ).sum().item()

    total_norm = total_squared ** 0.5

    assert total_norm <= 1.0