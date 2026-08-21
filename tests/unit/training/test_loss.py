import torch
import torch.nn.functional as F

from llm_systems.training.loss import cross_entropy_loss

def test_cross_entropy_simple():
    logits = torch.tensor(
        [[2.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [1],
        dtype=torch.long,
    )

    actual = cross_entropy_loss(
        logits,
        targets,
    )

    expected = F.cross_entropy(
        logits,
        targets,
    )

    torch.testing.assert_close(
        actual,
        expected,
    )

def test_cross_entropy_batch_sequence():
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0],
                [0.5, 2.5, 1.0],
            ],
            [
                [1.0, 0.0, 3.0],
                [2.0, 4.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            [0, 1],
            [2, 1],
        ],
        dtype=torch.long,
    )

    actual = cross_entropy_loss(
        logits,
        targets,
    )

    # PyTorch expects [N, C], so flatten B,T together.
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )

    torch.testing.assert_close(
        actual,
        expected,
    )

def test_cross_entropy_numerical_stability():
    logits = torch.tensor(
        [
            [1000.0, 999.0, 998.0],
            [10000.0, 10001.0, 9999.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    actual = cross_entropy_loss(
        logits,
        targets,
    )

    expected = F.cross_entropy(
        logits,
        targets,
    )

    assert torch.isfinite(actual)

    torch.testing.assert_close(
        actual,
        expected,
    )

def test_cross_entropy_arbitrary_batch_dimensions():
    torch.manual_seed(0)

    logits = torch.randn(
        2,
        3,
        4,
        7,
        dtype=torch.float32,
    )

    targets = torch.randint(
        low=0,
        high=7,
        size=(2, 3, 4),
        dtype=torch.long,
    )

    actual = cross_entropy_loss(
        logits,
        targets,
    )

    expected = F.cross_entropy(
        logits.reshape(-1, 7),
        targets.reshape(-1),
    )

    torch.testing.assert_close(
        actual,
        expected,
    )

def test_correct_prediction_has_lower_loss():
    targets = torch.tensor(
        [1],
        dtype=torch.long,
    )

    bad_logits = torch.tensor(
        [[10.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    good_logits = torch.tensor(
        [[0.0, 10.0, 0.0]],
        dtype=torch.float32,
    )

    bad_loss = cross_entropy_loss(
        bad_logits,
        targets,
    )

    good_loss = cross_entropy_loss(
        good_logits,
        targets,
    )

    print("bad_loss:", bad_loss)
    print("good_loss:", good_loss)

    assert good_loss < bad_loss