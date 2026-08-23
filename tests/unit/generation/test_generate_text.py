
import pytest
import torch

from llm_systems.generation.decoding import (
    sample_text_token,
    generate
)


def test_sample_token_is_valid():
    logits = torch.tensor([
        1.0,
        2.0,
        3.0,
    ])

    token = sample_text_token(
        logits,
        temperature=1.0,
    )

    assert token.numel() == 1
    assert 0 <= token.item() < len(logits)


def test_invalid_temperature():
    logits = torch.tensor([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        sample_text_token(
            logits,
            temperature=0.0,
        )

    with pytest.raises(ValueError):
        sample_text_token(
            logits,
            temperature=-1.0,
        )


def test_invalid_top_p():
    logits = torch.tensor([
        1.0,
        2.0,
        3.0,
    ])

    invalid_values = [
        0.0,
        -0.5,
        1.1,
    ]

    for top_p in invalid_values:
        with pytest.raises(ValueError):
            sample_text_token(
                logits,
                temperature=1.0,
                top_p=top_p,
            )


def test_top_p_keeps_highest_probability_token():
    # Token 0 overwhelmingly dominates.
    logits = torch.tensor([
        20.0,
        -20.0,
        -20.0,
    ])

    for _ in range(20):
        token = sample_text_token(
            logits,
            temperature=1.0,
            top_p=0.5,
        )

        assert token.item() == 0


def test_top_p_maps_back_to_original_token_id():
    # Highest logit is deliberately at vocab index 2,
    # not index 0.
    logits = torch.tensor([
        -20.0,
        -20.0,
        20.0,
        -20.0,
    ])

    for _ in range(20):
        token = sample_text_token(
            logits,
            temperature=1.0,
            top_p=0.5,
        )

        assert token.item() == 2


def test_top_p_one_allows_full_distribution():
    torch.manual_seed(0)

    logits = torch.tensor([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    token = sample_text_token(
        logits,
        temperature=1.0,
        top_p=1.0,
    )

    assert 0 <= token.item() < len(logits)

class DummyTokenizer:
    def encode(self, text):
        return [0]

    def decode(self, ids):
        return " ".join(
            str(i) for i in ids
        )

class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Gives model a parameter so generate()
        # can determine its device.
        self.dummy = torch.nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):

        B, T = x.shape

        logits = torch.full(
            (B, T, 3),
            -20.0,
            device=x.device,
        )

        # Always predict token 2
        logits[:, :, 2] = 20.0

        return logits

def test_generate_max_tokens():

    model = DummyModel()
    tokenizer = DummyTokenizer()

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=3,
        context_length=4,
        top_p=0.5,
    )

    assert text == "0 2 2 2"

def test_generate_stops_at_eos():

    model = DummyModel()
    tokenizer = DummyTokenizer()

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=10,
        context_length=4,
        top_p=0.5,
        eos_token_id=2,
    )

    assert text == "0 2"

def test_generate_restores_training_mode():

    model = DummyModel()
    tokenizer = DummyTokenizer()

    model.train()

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=1,
        context_length=4,
    )

    assert model.training