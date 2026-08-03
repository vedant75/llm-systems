import torch

from llm_systems.nn.transformer_block import TransformerBlock


def test_transformer_block_output_shape() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16

    block = TransformerBlock(
        d_model=d_model,
        num_heads=4,
        d_ff=32,
        theta=10_000.0,
        max_seq_len=32,
        dtype=torch.float32,
    )

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    token_positions = torch.arange(
        seq_len,
        dtype=torch.long,
    ).expand(
        batch_size,
        seq_len,
    )

    output = block(
        x,
        token_positions,
    )

    assert output.shape == x.shape