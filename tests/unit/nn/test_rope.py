import torch

from llm_systems.nn.rope import RotaryPositionalEmbedding


def test_rope_has_no_trainable_parameters() -> None:
    rope = RotaryPositionalEmbedding(
        theta=10_000.0,
        d_k=8,
        max_seq_len=32,
    )

    # RoPE itself should not learn anything.
    assert list(rope.parameters()) == []

    # Precomputed sin/cos tables should be buffers.
    buffers = dict(rope.named_buffers())

    assert "cos" in buffers
    assert "sin" in buffers

    assert buffers["cos"].shape == (32, 4)
    assert buffers["sin"].shape == (32, 4)


def test_rope_position_zero_is_identity() -> None:
    torch.manual_seed(42)

    d_k = 8

    rope = RotaryPositionalEmbedding(
        theta=10_000.0,
        d_k=d_k,
        max_seq_len=16,
    )

    x = torch.randn(
        2,
        3,
        d_k,
        dtype=torch.float32,
    )

    # Every token is assigned position 0.
    token_positions = torch.zeros(
        2,
        3,
        dtype=torch.long,
    )

    output = rope(
        x,
        token_positions,
    )

    torch.testing.assert_close(
        output,
        x,
    )


def test_rope_matches_manual_rotation() -> None:
    theta = 10_000.0
    d_k = 4

    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=8,
    )

    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )

    token_positions = torch.tensor(
        [[1]],
        dtype=torch.long,
    )

    output = rope(
        x,
        token_positions,
    )

    # Pair frequencies:
    #
    # pair 0: theta^(-0/4)
    # pair 1: theta^(-2/4)
    frequencies = theta ** (
        -torch.arange(
            0,
            d_k,
            2,
            dtype=torch.float32,
        )
        / d_k
    )

    angles = frequencies * 1.0

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Original pairs:
    #
    # (1, 2)
    # (3, 4)

    x1 = torch.tensor(
        [1.0, 3.0],
    )

    x2 = torch.tensor(
        [2.0, 4.0],
    )

    rotated_x1 = (
        x1 * cos
        - x2 * sin
    )

    rotated_x2 = (
        x1 * sin
        + x2 * cos
    )

    expected = torch.stack(
        [
            rotated_x1,
            rotated_x2,
        ],
        dim=-1,
    ).flatten()

    expected = expected.reshape(
        1,
        1,
        d_k,
    )

    torch.testing.assert_close(
        output,
        expected,
    )


def test_rope_preserves_vector_norm() -> None:
    torch.manual_seed(42)

    d_k = 8
    seq_len = 6

    rope = RotaryPositionalEmbedding(
        theta=10_000.0,
        d_k=d_k,
        max_seq_len=32,
    )

    x = torch.randn(
        2,
        seq_len,
        d_k,
        dtype=torch.float32,
    )

    token_positions = torch.arange(
        seq_len,
        dtype=torch.long,
    ).expand(2, -1)

    output = rope(
        x,
        token_positions,
    )

    input_norm = torch.linalg.vector_norm(
        x,
        dim=-1,
    )

    output_norm = torch.linalg.vector_norm(
        output,
        dim=-1,
    )

    torch.testing.assert_close(
        output_norm,
        input_norm,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rope_preserves_shape_and_dtype() -> None:
    d_k = 8
    seq_len = 5

    rope = RotaryPositionalEmbedding(
        theta=10_000.0,
        d_k=d_k,
        max_seq_len=32,
    )

    x = torch.randn(
        2,
        3,
        seq_len,
        d_k,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    # x is [batch, heads, seq_len, d_k]
    #
    # Therefore positions follow the assignment's
    # (..., seq_len) contract:
    # [batch, heads, seq_len]
    token_positions = torch.arange(
        seq_len,
        dtype=torch.long,
    )

    token_positions = token_positions.reshape(
        1,
        1,
        seq_len,
    ).expand(
        2,
        3,
        seq_len,
    )

    output = rope(
        x,
        token_positions,
    )

    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_rope_backward_is_finite() -> None:
    torch.manual_seed(42)

    d_k = 8
    seq_len = 5

    rope = RotaryPositionalEmbedding(
        theta=10_000.0,
        d_k=d_k,
        max_seq_len=16,
    )

    x = torch.randn(
        2,
        seq_len,
        d_k,
        dtype=torch.float64,
        requires_grad=True,
    )

    token_positions = torch.arange(
        seq_len,
        dtype=torch.long,
    ).expand(
        2,
        seq_len,
    )

    output = rope(
        x,
        token_positions,
    )

    upstream_gradient = torch.randn_like(
        output
    )

    output.backward(
        upstream_gradient
    )

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()