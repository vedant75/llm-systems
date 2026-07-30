import torch
import torch.nn.functional as F
from torch import nn

from llm_systems.nn.embedding import Embedding


def test_embedding_parameter_structure() -> None:
    embedding = Embedding(
        num_embeddings=20,
        embedding_dim=8,
    )

    # The embedding matrix must be trainable.
    assert isinstance(embedding.weight, nn.Parameter)

    # One row per vocabulary item and one column per embedding feature.
    assert embedding.weight.shape == (20, 8)

    # The checkpoint should contain the expected parameter name.
    assert list(embedding.state_dict()) == ["weight"]

    number_of_parameters = sum(
        parameter.numel()
        for parameter in embedding.parameters()
    )

    assert number_of_parameters == 20 * 8


def test_embedding_forward_matches_torch() -> None:
    torch.manual_seed(42)

    embedding = Embedding(
        num_embeddings=20,
        embedding_dim=8,
        dtype=torch.float64,
    )

    token_ids = torch.tensor(
        [
            [2, 4, 7, 2],
            [1, 9, 4, 3],
        ],
        dtype=torch.long,
    )

    custom_output = embedding(token_ids)

    reference_output = F.embedding(
        token_ids,
        embedding.weight,
    )

    assert custom_output.shape == (2, 4, 8)

    torch.testing.assert_close(
        custom_output,
        reference_output,
    )


def test_embedding_weight_gradient_matches_torch() -> None:
    torch.manual_seed(42)

    embedding = Embedding(
        num_embeddings=20,
        embedding_dim=8,
        dtype=torch.float64,
    )

    token_ids = torch.tensor(
        [
            [2, 2, 7, 4],
            [7, 2, 9, 4],
        ],
        dtype=torch.long,
    )

    # A separate weight tensor is used by the PyTorch reference path.
    reference_weight = (
        embedding.weight
        .detach()
        .clone()
        .requires_grad_(True)
    )

    custom_output = embedding(token_ids)

    reference_output = F.embedding(
        token_ids,
        reference_weight,
    )

    upstream_gradient = torch.randn_like(custom_output)

    custom_output.backward(upstream_gradient)
    reference_output.backward(upstream_gradient)

    torch.testing.assert_close(
        embedding.weight.grad,
        reference_weight.grad,
    )


def test_embedding_accumulates_repeated_token_gradients() -> None:
    embedding = Embedding(
        num_embeddings=10,
        embedding_dim=4,
        dtype=torch.float64,
    )

    token_ids = torch.tensor(
        [2, 5, 2],
        dtype=torch.long,
    )

    output = embedding(token_ids)

    # output.sum() gives an upstream gradient of one
    # for every embedding output element.
    output.sum().backward()

    expected_gradient = torch.zeros_like(
        embedding.weight
    )

    # Token 2 appears twice.
    expected_gradient[2] = 2.0

    # Token 5 appears once.
    expected_gradient[5] = 1.0

    torch.testing.assert_close(
        embedding.weight.grad,
        expected_gradient,
    )


def test_embedding_supports_arbitrary_input_shapes() -> None:
    embedding = Embedding(
        num_embeddings=20,
        embedding_dim=8,
    )

    scalar_token = torch.tensor(
        3,
        dtype=torch.long,
    )

    token_sequence = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    token_batch = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=torch.long,
    )

    assert embedding(scalar_token).shape == (8,)
    assert embedding(token_sequence).shape == (3, 8)
    assert embedding(token_batch).shape == (2, 2, 8)


def test_embedding_initialization_bounds() -> None:
    embedding = Embedding(
        num_embeddings=128,
        embedding_dim=64,
    )

    assert torch.isfinite(embedding.weight).all()

    assert torch.all(
        embedding.weight >= -3.0
    )

    assert torch.all(
        embedding.weight <= 3.0
    )