import torch
from torch import nn

from llm_systems.nn.rmsnorm import RMSNorm


def test_rmsnorm_parameter_structure() -> None:
    norm = RMSNorm(
        d_model=8,
        eps=1e-5,
    )

    assert norm.d_model == 8
    assert norm.eps == 1e-5

    # RMSNorm has one learnable gain value per model feature.
    assert isinstance(norm.weight, nn.Parameter)
    assert norm.weight.shape == (8,)

    # The gain should initially contain ones.
    torch.testing.assert_close(
        norm.weight,
        torch.ones(8),
    )

    assert list(norm.state_dict()) == ["weight"]

    number_of_parameters = sum(
        parameter.numel()
        for parameter in norm.parameters()
    )

    assert number_of_parameters == 8


def test_rmsnorm_matches_manual_equation() -> None:
    torch.manual_seed(42)

    d_model = 8
    eps = 1e-5

    norm = RMSNorm(
        d_model=d_model,
        eps=eps,
        dtype=torch.float32,
    )

    # Use non-uniform gain values so the test also checks
    # feature-wise gain multiplication.
    with torch.no_grad():
        norm.weight.copy_(
            torch.linspace(
                0.5,
                1.5,
                steps=d_model,
            )
        )

    x = torch.randn(
        2,
        5,
        d_model,
        dtype=torch.float32,
    )

    custom_output = norm(x)

    x_float = x.to(torch.float32)

    mean_square = x_float.square().mean(
        dim=-1,
        keepdim=True,
    )

    expected_output = (
        x_float
        * torch.rsqrt(mean_square + eps)
        * norm.weight
    )

    expected_output = expected_output.to(x.dtype)

    assert custom_output.shape == x.shape

    torch.testing.assert_close(
        custom_output,
        expected_output,
    )


def test_rmsnorm_forward_and_backward_match_torch() -> None:
    torch.manual_seed(42)

    d_model = 8
    eps = 1e-5

    custom_norm = RMSNorm(
        d_model=d_model,
        eps=eps,
        dtype=torch.float32,
    )

    reference_norm = nn.RMSNorm(
        normalized_shape=d_model,
        eps=eps,
        elementwise_affine=True,
        dtype=torch.float32,
    )

    gain = torch.linspace(
        0.5,
        1.5,
        steps=d_model,
    )

    with torch.no_grad():
        custom_norm.weight.copy_(gain)
        reference_norm.weight.copy_(gain)

    custom_input = torch.randn(
        2,
        5,
        d_model,
        dtype=torch.float32,
        requires_grad=True,
    )

    reference_input = (
        custom_input
        .detach()
        .clone()
        .requires_grad_(True)
    )

    custom_output = custom_norm(custom_input)
    reference_output = reference_norm(reference_input)

    torch.testing.assert_close(
        custom_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6,
    )

    upstream_gradient = torch.randn_like(
        custom_output
    )

    custom_output.backward(upstream_gradient)
    reference_output.backward(upstream_gradient)

    # Gradient flowing back into the previous layer.
    torch.testing.assert_close(
        custom_input.grad,
        reference_input.grad,
        rtol=1e-5,
        atol=1e-6,
    )

    # Gradient used to update the learnable gain.
    torch.testing.assert_close(
        custom_norm.weight.grad,
        reference_norm.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rmsnorm_preserves_shape_and_input_dtype() -> None:
    norm = RMSNorm(
        d_model=8,
        dtype=torch.bfloat16,
    )

    x = torch.randn(
        2,
        5,
        8,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    output = norm(x)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()


def test_rmsnorm_zero_input_is_finite() -> None:
    norm = RMSNorm(
        d_model=8,
        eps=1e-5,
    )

    x = torch.zeros(
        2,
        5,
        8,
    )

    output = norm(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    torch.testing.assert_close(
        output,
        torch.zeros_like(output),
    )


def test_rmsnorm_normalizes_only_final_dimension() -> None:
    norm = RMSNorm(
        d_model=8,
        eps=1e-5,
    )

    vector_input = torch.randn(8)
    sequence_input = torch.randn(4, 8)
    batch_input = torch.randn(2, 5, 8)

    assert norm(vector_input).shape == (8,)
    assert norm(sequence_input).shape == (4, 8)
    assert norm(batch_input).shape == (2, 5, 8)