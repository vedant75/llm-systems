import math

import torch
import torch.nn.functional as F
from torch import nn

from llm_systems.nn.linear import Linear


def test_linear_parameter_structure() -> None:
    layer = Linear(
        in_features=8,
        out_features=16,
    )

    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (16, 8)
    assert list(layer.state_dict()) == ["weight"]

    number_of_parameters = sum(
        parameter.numel()
        for parameter in layer.parameters()
    )

    assert number_of_parameters == 8 * 16


def test_linear_forward_matches_torch() -> None:
    torch.manual_seed(42)

    layer = Linear(
        in_features=8,
        out_features=16,
        dtype=torch.float64,
    )

    x = torch.randn(
        2,
        5,
        8,
        dtype=torch.float64,
    )

    actual = layer(x)

    expected = F.linear(
        x,
        layer.weight,
        bias=None,
    )

    assert actual.shape == (2, 5, 16)

    torch.testing.assert_close(
        actual,
        expected,
    )


def test_linear_backward_matches_torch() -> None:
    torch.manual_seed(42)

    layer = Linear(
        in_features=8,
        out_features=16,
        dtype=torch.float64,
    )

    x_custom = torch.randn(
        2,
        5,
        8,
        dtype=torch.float64,
        requires_grad=True,
    )

    x_reference = (
        x_custom.detach()
        .clone()
        .requires_grad_(True)
    )

    custom_output = layer(x_custom)

    reference_output = F.linear(
        x_reference,
        layer.weight,
        bias=None,
    )

    upstream_gradient = torch.randn_like(custom_output)

    custom_output.backward(upstream_gradient)

    custom_input_gradient = x_custom.grad.clone()
    custom_weight_gradient = layer.weight.grad.clone()

    layer.weight.grad = None

    reference_output.backward(upstream_gradient)

    torch.testing.assert_close(
        custom_input_gradient,
        x_reference.grad,
    )

    torch.testing.assert_close(
        custom_weight_gradient,
        layer.weight.grad,
    )


def test_linear_supports_arbitrary_leading_dimensions() -> None:
    layer = Linear(
        in_features=8,
        out_features=16,
    )

    assert layer(torch.randn(8)).shape == (16,)
    assert layer(torch.randn(4, 8)).shape == (4, 16)
    assert layer(torch.randn(2, 5, 8)).shape == (2, 5, 16)
    assert layer(torch.randn(2, 4, 5, 8)).shape == (2, 4, 5, 16)


def test_linear_initialization_bounds() -> None:
    in_features = 64
    out_features = 128

    layer = Linear(
        in_features=in_features,
        out_features=out_features,
    )

    std = math.sqrt(
        2.0 / (in_features + out_features)
    )

    assert torch.isfinite(layer.weight).all()
    assert torch.all(layer.weight >= -3.0 * std)
    assert torch.all(layer.weight <= 3.0 * std)