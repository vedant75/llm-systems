import torch
import torch.nn.functional as F
from einops import rearrange

from llm_systems.nn.multiheadattention import MultiHeadAttention


def test_multihead_attention_parameter_shapes() -> None:
    d_model = 16
    num_heads = 4

    attention = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=10_000.0,
        max_seq_len=32,
    )

    assert attention.head_dim == 4

    assert attention.Wq.weight.shape == (
        d_model,
        d_model,
    )

    assert attention.Wk.weight.shape == (
        d_model,
        d_model,
    )

    assert attention.Wv.weight.shape == (
        d_model,
        d_model,
    )

    assert attention.Wo.weight.shape == (
        d_model,
        d_model,
    )


def test_multihead_attention_output_shape() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    attention = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=10_000.0,
        max_seq_len=32,
        dtype=torch.float64,
    )

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
        dtype=torch.float64,
    )

    token_positions = torch.arange(
        seq_len,
        dtype=torch.long,
    ).expand(
        batch_size,
        seq_len,
    )

    output = attention(
        x,
        token_positions,
    )

    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    )


def test_multihead_attention_matches_reference() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    head_dim = d_model // num_heads

    attention = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=10_000.0,
        max_seq_len=32,
        dtype=torch.float64,
    )

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
        dtype=torch.float64,
    )

    token_positions = torch.arange(
        seq_len,
        dtype=torch.long,
    ).expand(
        batch_size,
        seq_len,
    )

    # ---------------------------------------------
    # Custom implementation
    # ---------------------------------------------

    custom_output = attention(
        x,
        token_positions,
    )

    # ---------------------------------------------
    # Independent reference path
    # ---------------------------------------------

    q = F.linear(
        x,
        attention.Wq.weight,
        bias=None,
    )

    k = F.linear(
        x,
        attention.Wk.weight,
        bias=None,
    )

    v = F.linear(
        x,
        attention.Wv.weight,
        bias=None,
    )

    q = rearrange(
        q,
        "b seq (head dim) -> b head seq dim",
        head=num_heads,
    )

    k = rearrange(
        k,
        "b seq (head dim) -> b head seq dim",
        head=num_heads,
    )

    v = rearrange(
        v,
        "b seq (head dim) -> b head seq dim",
        head=num_heads,
    )

    rope_positions = token_positions.unsqueeze(1)

    q = attention.rope(
        q,
        rope_positions,
    )

    k = attention.rope(
        k,
        rope_positions,
    )

    causal_mask = torch.ones(
        seq_len,
        seq_len,
        dtype=torch.bool,
    ).tril()

    reference_attention = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=False,
    )

    reference_attention = rearrange(
        reference_attention,
        "b head seq dim -> b seq (head dim)",
    )

    reference_output = F.linear(
        reference_attention,
        attention.Wo.weight,
        bias=None,
    )

    assert custom_output.shape == reference_output.shape

    torch.testing.assert_close(
        custom_output,
        reference_output,
        rtol=1e-6,
        atol=1e-6,
    )


def test_multihead_attention_is_causal() -> None:
    torch.manual_seed(42)

    seq_len = 6
    d_model = 16

    attention = MultiHeadAttention(
        d_model=d_model,
        num_heads=4,
        theta=10_000.0,
        max_seq_len=32,
        dtype=torch.float64,
    )

    original_x = torch.randn(
        1,
        seq_len,
        d_model,
        dtype=torch.float64,
    )

    modified_x = original_x.clone()

    # Change ONLY the final token substantially.
    modified_x[:, -1, :] += 100.0

    token_positions = torch.arange(
        seq_len,
        dtype=torch.long,
    ).unsqueeze(0)

    original_output = attention(
        original_x,
        token_positions,
    )

    modified_output = attention(
        modified_x,
        token_positions,
    )

    # Earlier tokens must not be affected by a future token.
    torch.testing.assert_close(
        original_output[:, :-1, :],
        modified_output[:, :-1, :],
        rtol=1e-6,
        atol=1e-6,
    )

    # The final token itself is allowed to change.
    assert not torch.allclose(
        original_output[:, -1, :],
        modified_output[:, -1, :],
    )