from __future__ import annotations

import math

import torch
from jaxtyping import Float

from einops import einsum


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q: Float[torch.Tensor, "B seq d"],
        K: Float[torch.Tensor, "B seq d"],
        V: Float[torch.Tensor, "B seq d"],
        is_causal: bool = False,
    ):
        B, Tq, d = Q.shape
        Tk = K.shape[-2]

        Q_tile_size = 16
        K_tile_size = 16

        # states
        L = torch.zeros(
            (B, Tq),
            device=Q.device,
            dtype=Q.dtype,
        )

        O = torch.zeros(
            (B, Tq, d),
            device=Q.device,
            dtype=Q.dtype,
        )

        for b in range(B):
            for tile in range(0, Tq, Q_tile_size):
                q_end = min(tile + Q_tile_size, Tq)

                Q_i = Q[b, tile:q_end, :]

                Bq = Q_i.shape[-2]
                
                m = torch.full(
                    (Bq, 1),
                    float("-inf"),
                    device=Q.device,
                    dtype=Q.dtype,
                )

                l = torch.zeros(
                    (Bq, 1),
                    device=Q.device,
                    dtype=Q.dtype,
                )

                O_accu = torch.zeros(
                    (Bq, d),
                    device=Q.device,
                    dtype=Q.dtype,
                )

                for k_tile in range(0, Tk, K_tile_size):
                    k_end = min(k_tile + K_tile_size, Tk)

                    K_j = K[b, k_tile:k_end, :]
                    V_j = V[b, k_tile:k_end, :]

                    S = Q_i @ K_j.transpose(-2, -1)
                    S = S / math.sqrt(d)

                    if is_causal:
                        q_positions = torch.arange(
                            tile,
                            q_end,
                            device=Q.device,
                        ).unsqueeze(1)

                        k_positions = torch.arange(
                            k_tile,
                            k_end,
                            device=Q.device,
                        ).unsqueeze(0)

                        causal_mask = k_positions > q_positions

                        S = S.masked_fill(
                            causal_mask,
                            float("-inf"),
                        )

                    m_tile = torch.max(
                        S,
                        dim=-1,
                        keepdim=True
                    ).values

                    m_new = torch.maximum(
                        m,
                        m_tile,
                    )

                    alpha = torch.exp(
                        m - m_new
                    )

                    p_tilda = torch.exp(
                        S - m_new
                    )

                    l_new = (
                        alpha * l
                        + torch.sum(
                            p_tilda,
                            dim=-1,
                            keepdim=True,
                        )
                    )

                    O_new = (
                        alpha * O_accu
                        + p_tilda @ V_j
                    )

                    m = m_new
                    l = l_new
                    O_accu = O_new

                O[b, tile:q_end, :] = O_accu / l

                val = m + torch.log(l)
                L[b, tile:q_end] = val.squeeze(-1)
        
        ctx.save_for_backward(
            Q,
            K,
            V,
            O,
            L,
        )

        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        Q, K, V, O, L = ctx.saved_tensors

        B, Tq, d = Q.shape
        Tk = K.shape[-2]

        Q_tile_size = 16
        K_tile_size = 16

        scale = 1.0 / math.sqrt(d)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # D_i = sum_k O_ik * dO_ik
        D = torch.sum(
            O * grad_output,
            dim=-1,
            keepdim=True,
        )

        for b in range(B):
            for q_tile in range(0, Tq, Q_tile_size):
                q_end = min(
                    q_tile + Q_tile_size,
                    Tq,
                )

                Q_i = Q[b, q_tile:q_end, :]
                dO_i = grad_output[b, q_tile:q_end, :]

                L_i = L[b, q_tile:q_end].unsqueeze(-1)
                D_i = D[b, q_tile:q_end, :]

                dQ_accu = torch.zeros_like(Q_i)

                for k_tile in range(0, Tk, K_tile_size):
                    k_end = min(
                        k_tile + K_tile_size,
                        Tk,
                    )

                    K_j = K[b, k_tile:k_end, :]
                    V_j = V[b, k_tile:k_end, :]

                    # Recompute attention scores.
                    S = Q_i @ K_j.transpose(-2, -1)
                    S = S * scale

                    if ctx.is_causal:
                        q_positions = torch.arange(
                            q_tile,
                            q_end,
                            device=Q.device,
                        ).unsqueeze(1)

                        k_positions = torch.arange(
                            k_tile,
                            k_end,
                            device=Q.device,
                        ).unsqueeze(0)

                        causal_mask = k_positions > q_positions

                        S = S.masked_fill(
                            causal_mask,
                            float("-inf"),
                        )

                    # Reconstruct softmax probabilities.
                    P = torch.exp(
                        S - L_i
                    )

                    # Gradient with respect to V.
                    dV[b, k_tile:k_end, :] += (
                        P.transpose(-2, -1) @ dO_i
                    )

                    # Gradient with respect to P.
                    dP = (
                        dO_i
                        @ V_j.transpose(-2, -1)
                    )

                    # Softmax backward.
                    dS = P * (
                        dP - D_i
                    )

                    # Gradient with respect to Q.
                    dQ_accu += (
                        dS @ K_j
                    ) * scale

                    # Gradient with respect to K.
                    dK[b, k_tile:k_end, :] += (
                        dS.transpose(-2, -1)
                        @ Q_i
                    ) * scale

                dQ[b, q_tile:q_end, :] = dQ_accu

        return dQ, dK, dV, None