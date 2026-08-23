import torch

from llm_systems.nn.attention import softmax


def sample_text_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError(
            f'Error Provided temperature Value: {temperature}'
        )

    scaled_logits = logits / temperature

    probs = softmax(
        scaled_logits,
        dim=-1
    )

    if top_p is None:
        next_token = torch.multinomial(
            probs,
            num_samples= 1
        )
    
    else:
        if not 0 < top_p <= 1:
            raise ValueError(
                f"top_p must be in (0, 1], "
                f"got {top_p}"
            )
        # sorted probs
        sorted_probs, sorted_indices = torch.sort(
            probs,
            descending= True
        )

        # calculate cumsum
        cumsum_probs = torch.cumsum(
            sorted_probs,
            dim= -1
        )

        # mask
        mask = cumsum_probs > top_p

        # shift mask
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        # zero out filtered probs
        sorted_probs.masked_fill_(
            mask,
            0.0
        )

        # normalize
        sorted_probs = (
            sorted_probs / sorted_probs.sum(dim=-1, keepdim= True)
        )

        # sample distribution
        sampled_position = torch.multinomial(
            sorted_probs,
            num_samples=1,
        )

        # next token
        next_token = torch.gather(
            sorted_indices,
            dim=-1,
            index=sampled_position,
        )
    
    return next_token


def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> str:

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be >= 0"
        )

    if context_length <= 0:
        raise ValueError(
            "context_length must be > 0"
        )

    token_ids = tokenizer.encode(prompt)

    if len(token_ids) == 0:
        raise ValueError(
            "Prompt must contain at least one token."
        )

    device = next(
        model.parameters()
    ).device

    was_training = model.training

    model.eval()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Only give the model as much history
            # as its context window supports.
            model_input_ids = token_ids[
                -context_length:
            ]

            x = torch.tensor(
                model_input_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            # [1, T] -> [1, T, V]
            logits = model(x)

            # We only need the prediction
            # after the final current token.
            next_token_logits = logits[0, -1, :]

            next_token = sample_text_token(
                logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
            )

            next_token_id = next_token.item()

            token_ids.append(
                next_token_id
            )

            if (
                eos_token_id is not None
                and next_token_id
                == eos_token_id
            ):
                break

    if was_training:
        model.train()

    return tokenizer.decode(
        token_ids
    )