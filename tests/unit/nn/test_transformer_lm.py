import pytest
import torch
from torch import nn

from llm_systems.nn.transformer_lm import TransformerLM


def test_transformer_lm_structure() -> None:
    vocab_size = 100
    d_model = 16
    num_layers = 3

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=32,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=4,
        d_ff=32,
        theta=10_000.0,
    )

    assert isinstance(
        model.layers,
        nn.ModuleList,
    )

    assert len(model.layers) == num_layers

    # Every block should be its own module.
    assert model.layers[0] is not model.layers[1]
    assert model.layers[1] is not model.layers[2]

    assert model.token_embedding.weight.shape == (
        vocab_size,
        d_model,
    )

    assert model.lm_head.weight.shape == (
        vocab_size,
        d_model,
    )