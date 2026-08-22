import torch

from llm_systems.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

def test_checkpoint(tmp_path):
    torch.manual_seed(0)

    # -------------------------
    # Original model/optimizer
    # -------------------------

    model = torch.nn.Linear(3, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # Perform one training step so the optimizer
    # actually has internal Adam state.
    x = torch.randn(4, 3)

    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    iteration = 42

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    # Save copies of the trained parameters
    expected_parameters = {
        name: tensor.clone()
        for name, tensor
        in model.state_dict().items()
    }

    # -------------------------
    # Save
    # -------------------------

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=checkpoint_path,
    )

    # -------------------------
    # Create fresh objects
    # -------------------------

    new_model = torch.nn.Linear(3, 2)

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=1e-3,
    )

    # -------------------------
    # Load
    # -------------------------

    loaded_iteration = load_checkpoint(
        src=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
    )

    # -------------------------
    # Check iteration
    # -------------------------

    assert loaded_iteration == iteration

    # -------------------------
    # Check model weights
    # -------------------------

    for name, tensor in new_model.state_dict().items():
        torch.testing.assert_close(
            tensor,
            expected_parameters[name],
        )

    # -------------------------
    # Check optimizer state
    # -------------------------

    original_state = optimizer.state_dict()
    loaded_state = new_optimizer.state_dict()

    assert (
        original_state["param_groups"]
        == loaded_state["param_groups"]
    )

    assert (
        original_state["state"].keys()
        == loaded_state["state"].keys()
    )