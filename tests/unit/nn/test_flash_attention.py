import torch

from llm_systems.nn.attention import scaled_dot_product_attention
from llm_systems.nn.flash_attention import FlashAttention


def test_flash_attention_forward():
    torch.manual_seed(42)

    B = 1
    T = 32
    d = 8

    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)

    expected = scaled_dot_product_attention(
        Q=Q,
        K=K,
        V=V,
        mask=None,
    )

    actual = FlashAttention.apply(
        Q,
        K,
        V,
        False,
    )

    assert actual.shape == expected.shape

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

def test_flash_attention_forward2():
    torch.manual_seed(42)

    B = 2
    T = 32
    d = 16

    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)

    expected = scaled_dot_product_attention(
        Q=Q,
        K=K,
        V=V,
        mask=None,
    )

    actual = FlashAttention.apply(
        Q,
        K,
        V,
        False,
    )

    assert actual.shape == expected.shape

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )



def test_flash_attention_forward3():
    torch.manual_seed(42)

    B = 2
    T = 37
    d = 16

    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)

    expected = scaled_dot_product_attention(
        Q=Q,
        K=K,
        V=V,
        mask=None,
    )

    actual = FlashAttention.apply(
        Q,
        K,
        V,
        False,
    )

    assert actual.shape == expected.shape

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

def test_flash_attention_forward4():
    torch.manual_seed(42)

    B = 2
    T = 32
    d = 64

    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)

    expected = scaled_dot_product_attention(
        Q=Q,
        K=K,
        V=V,
        mask=None,
    )

    actual = FlashAttention.apply(
        Q,
        K,
        V,
        False,
    )

    assert actual.shape == expected.shape

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

# forward with casual mask
def test_flash_attention_causal_forward():
    torch.manual_seed(42)

    B = 2
    T = 32
    d = 16

    Q = torch.randn(B, T, d)
    K = torch.randn(B, T, d)
    V = torch.randn(B, T, d)

    scores = (
        Q @ K.transpose(-2, -1)
    ) / (d ** 0.5)

    causal_mask = torch.triu(
        torch.ones(
            T,
            T,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    scores = scores.masked_fill(
        causal_mask,
        float("-inf"),
    )

    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    expected = probabilities @ V

    actual = FlashAttention.apply(
        Q,
        K,
        V,
        True,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


# Backward pass

def test_flash_attention_backward():
    torch.manual_seed(42)

    B = 1
    T = 32
    d = 8

    Q_ref = torch.randn(B, T, d, requires_grad=True)
    K_ref = torch.randn(B, T, d, requires_grad=True)
    V_ref = torch.randn(B, T, d, requires_grad=True)

    Q_flash = Q_ref.detach().clone().requires_grad_(True)
    K_flash = K_ref.detach().clone().requires_grad_(True)
    V_flash = V_ref.detach().clone().requires_grad_(True)

    expected = scaled_dot_product_attention(
        Q=Q_ref,
        K=K_ref,
        V=V_ref,
        mask=None,
    )

    actual = FlashAttention.apply(
        Q_flash,
        K_flash,
        V_flash,
        False,
    )

    grad_output = torch.randn_like(expected)

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        Q_flash.grad,
        Q_ref.grad,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        K_flash.grad,
        K_ref.grad,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        V_flash.grad,
        V_ref.grad,
        rtol=1e-5,
        atol=1e-5,
    )

# backward with casual mask
def test_flash_attention_causal_backward():
    torch.manual_seed(42)

    B = 2
    T = 32
    d = 16

    Q_ref = torch.randn(
        B,
        T,
        d,
        requires_grad=True,
    )

    K_ref = torch.randn(
        B,
        T,
        d,
        requires_grad=True,
    )

    V_ref = torch.randn(
        B,
        T,
        d,
        requires_grad=True,
    )

    Q_flash = (
        Q_ref.detach()
        .clone()
        .requires_grad_(True)
    )

    K_flash = (
        K_ref.detach()
        .clone()
        .requires_grad_(True)
    )

    V_flash = (
        V_ref.detach()
        .clone()
        .requires_grad_(True)
    )

    scores = (
        Q_ref @ K_ref.transpose(-2, -1)
    ) / (d ** 0.5)

    causal_mask = torch.triu(
        torch.ones(
            T,
            T,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    scores = scores.masked_fill(
        causal_mask,
        float("-inf"),
    )

    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    expected = probabilities @ V_ref

    actual = FlashAttention.apply(
        Q_flash,
        K_flash,
        V_flash,
        True,
    )

    grad_output = torch.randn_like(expected)

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        Q_flash.grad,
        Q_ref.grad,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        K_flash.grad,
        K_ref.grad,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        V_flash.grad,
        V_ref.grad,
        rtol=1e-5,
        atol=1e-5,
    )