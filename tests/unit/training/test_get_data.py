# tests/unit/training/test_data_loader.py

import numpy as np
import torch

from llm_systems.training.get_data import get_data

def test_get_data_shapes():
    data = np.arange(100, dtype=np.int64)

    batch_size = 4
    context_length = 8

    inputs, targets = get_data(
        data=data,
        batch_size=batch_size,
        context_length=context_length,
        device="cpu",
    )

    assert inputs.shape == (batch_size, context_length)
    assert targets.shape == (batch_size, context_length)

def test_get_data_targets_are_shifted_by_one():
    data = np.arange(100, dtype=np.int64)

    inputs, targets = get_data(
        data=data,
        batch_size=8,
        context_length=5,
        device="cpu",
    )

    # Within every sampled sequence:
    #
    # input:
    # [10, 11, 12, 13, 14]
    #
    # target:
    # [11, 12, 13, 14, 15]
    #
    # Therefore input[:, 1:] must equal target[:, :-1].

    torch.testing.assert_close(
        inputs[:, 1:],
        targets[:, :-1],
    )

def test_get_data_dtype_and_device():
    data = np.arange(50, dtype=np.int32)

    inputs, targets = get_data(
        data=data,
        batch_size=2,
        context_length=4,
        device="cpu",
    )

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long

    assert inputs.device.type == "cpu"
    assert targets.device.type == "cpu"

def test_get_data_never_returns_short_sequences():
    data = np.arange(10, dtype=np.int64)

    batch_size = 100
    context_length = 4

    inputs, targets = get_data(
        data=data,
        batch_size=batch_size,
        context_length=context_length,
        device="cpu",
    )

    assert inputs.shape == (batch_size, context_length)
    assert targets.shape == (batch_size, context_length)

def test_get_data_samples_valid_contiguous_sequences():
    data = np.arange(30, dtype=np.int64)

    inputs, targets = get_data(
        data=data,
        batch_size=10,
        context_length=6,
        device="cpu",
    )

    for x, y in zip(inputs, targets):
        combined = torch.cat([
            x,
            y[-1:].clone(),
        ])

        # Because data = [0, 1, 2, 3, ...],
        # a valid T+1 slice must increase by exactly 1.
        expected = torch.arange(
            combined[0],
            combined[0] + len(combined),
            dtype=torch.long,
        )

        torch.testing.assert_close(
            combined,
            expected,
        )


