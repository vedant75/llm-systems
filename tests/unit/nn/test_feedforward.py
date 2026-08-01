import torch
import torch.nn.functional as F

from llm_systems.nn.feedforward import SwiGLU


def test_swiglu_forward_and_backward_matches_reference() -> None:
    torch.manual_seed(42)

    d_model = 8
    d_ff = 24

    swiglu = SwiGLU(
        d_model=d_model,
        d_ff=d_ff,
        dtype=torch.float64,
    )

    x_custom = torch.randn(
        2,
        5,
        d_model,
        dtype=torch.float64,
        requires_grad=True,
    )

    x_reference = (
        x_custom
        .detach()
        .clone()
        .requires_grad_(True)
    )

    # Make independent copies of the parameters for the reference path.
    w1_reference = (
        swiglu.w1.weight
        .detach()
        .clone()
        .requires_grad_(True)
    )

    w3_reference = (
        swiglu.w3.weight
        .detach()
        .clone()
        .requires_grad_(True)
    )

    w2_reference = (
        swiglu.w2.weight
        .detach()
        .clone()
        .requires_grad_(True)
    )

    # -----------------------
    # Custom implementation
    # -----------------------

    custom_output = swiglu(x_custom)

    # -----------------------
    # Reference implementation
    # -----------------------

    w1_x = F.linear(
        x_reference,
        w1_reference,
        bias=None,
    )

    gate = F.silu(w1_x)

    value = F.linear(
        x_reference,
        w3_reference,
        bias=None,
    )

    gated = gate * value

    reference_output = F.linear(
        gated,
        w2_reference,
        bias=None,
    )

    # -----------------------
    # Forward checks
    # -----------------------

    assert custom_output.shape == (
        2,
        5,
        d_model,
    )

    torch.testing.assert_close(
        custom_output,
        reference_output,
    )

    # -----------------------
    # Backward checks
    # -----------------------

    upstream_gradient = torch.randn_like(
        custom_output
    )

    custom_output.backward(
        upstream_gradient
    )

    reference_output.backward(
        upstream_gradient
    )

    # Gradient flowing back into the previous Transformer layer.
    torch.testing.assert_close(
        x_custom.grad,
        x_reference.grad,
    )

    # Gradients for all three SwiGLU projections.
    torch.testing.assert_close(
        swiglu.w1.weight.grad,
        w1_reference.grad,
    )

    torch.testing.assert_close(
        swiglu.w3.weight.grad,
        w3_reference.grad,
    )

    torch.testing.assert_close(
        swiglu.w2.weight.grad,
        w2_reference.grad,
    )