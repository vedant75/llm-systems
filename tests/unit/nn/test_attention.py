import torch
import torch.nn.functional as F

from llm_systems.nn.attention import (
    scaled_dot_product_attention,
    softmax,
)


def test_softmax_matches_torch() -> None:
    torch.manual_seed(42)

    x = torch.randn(
        3,
        4,
        5,
        dtype=torch.float64,
    )

    custom_output = softmax(
        x,
        dim=-1,
    )

    reference_output = torch.softmax(
        x,
        dim=-1,
    )

    assert custom_output.shape == x.shape

    torch.testing.assert_close(
        custom_output,
        reference_output,
    )


def test_softmax_probabilities_sum_to_one() -> None:
    torch.manual_seed(42)

    x = torch.randn(
        2,
        4,
        8,
        dtype=torch.float64,
    )

    probabilities = softmax(
        x,
        dim=-1,
    )

    probability_sums = probabilities.sum(
        dim=-1,
    )

    torch.testing.assert_close(
        probability_sums,
        torch.ones_like(probability_sums),
    )

    assert torch.all(probabilities >= 0)


def test_softmax_is_numerically_stable() -> None:
    x = torch.tensor(
        [
            [1000.0, 1001.0, 1002.0],
            [10000.0, 10001.0, 10002.0],
        ],
        dtype=torch.float64,
    )

    custom_output = softmax(
        x,
        dim=-1,
    )

    reference_output = torch.softmax(
        x,
        dim=-1,
    )

    assert torch.isfinite(custom_output).all()

    torch.testing.assert_close(
        custom_output,
        reference_output,
    )


def test_scaled_dot_product_attention_matches_torch() -> None:
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    query_length = 5
    key_length = 7
    d_k = 8
    d_v = 6

    q = torch.randn(
        batch_size,
        num_heads,
        query_length,
        d_k,
        dtype=torch.float64,
    )

    k = torch.randn(
        batch_size,
        num_heads,
        key_length,
        d_k,
        dtype=torch.float64,
    )

    v = torch.randn(
        batch_size,
        num_heads,
        key_length,
        d_v,
        dtype=torch.float64,
    )

    custom_output = scaled_dot_product_attention(
        q,
        k,
        v,
    )

    reference_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
    )

    assert custom_output.shape == (
        batch_size,
        num_heads,
        query_length,
        d_v,
    )

    torch.testing.assert_close(
        custom_output,
        reference_output,
        rtol=1e-6,
        atol=1e-6,
    )


def test_scaled_dot_product_attention_with_mask_matches_torch() -> None:
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 3
    sequence_length = 5
    d_k = 8
    d_v = 6

    q = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_k,
        dtype=torch.float64,
    )

    k = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_k,
        dtype=torch.float64,
    )

    v = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_v,
        dtype=torch.float64,
    )

    # True  -> allowed
    # False -> blocked
    mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        )
    )

    custom_output = scaled_dot_product_attention(
        q,
        k,
        v,
        mask=mask,
    )

    reference_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )

    torch.testing.assert_close(
        custom_output,
        reference_output,
        rtol=1e-6,
        atol=1e-6,
    )


def test_scaled_dot_product_attention_backward_matches_torch() -> None:
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 2
    sequence_length = 4
    d_k = 8
    d_v = 6

    q_custom = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_k,
        dtype=torch.float64,
        requires_grad=True,
    )

    k_custom = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_k,
        dtype=torch.float64,
        requires_grad=True,
    )

    v_custom = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        d_v,
        dtype=torch.float64,
        requires_grad=True,
    )

    q_reference = (
        q_custom
        .detach()
        .clone()
        .requires_grad_(True)
    )

    k_reference = (
        k_custom
        .detach()
        .clone()
        .requires_grad_(True)
    )

    v_reference = (
        v_custom
        .detach()
        .clone()
        .requires_grad_(True)
    )

    custom_output = scaled_dot_product_attention(
        q_custom,
        k_custom,
        v_custom,
    )

    reference_output = F.scaled_dot_product_attention(
        q_reference,
        k_reference,
        v_reference,
        dropout_p=0.0,
        is_causal=False,
    )

    upstream_gradient = torch.randn_like(
        custom_output
    )

    custom_output.backward(
        upstream_gradient
    )

    reference_output.backward(
        upstream_gradient
    )

    torch.testing.assert_close(
        q_custom.grad,
        q_reference.grad,
        rtol=1e-6,
        atol=1e-6,
    )

    torch.testing.assert_close(
        k_custom.grad,
        k_reference.grad,
        rtol=1e-6,
        atol=1e-6,
    )

    torch.testing.assert_close(
        v_custom.grad,
        v_reference.grad,
        rtol=1e-6,
        atol=1e-6,
    )