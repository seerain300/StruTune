import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_l, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, L, ceil_div(H, BLOCK_H))
    b = tl.program_id(0)
    m = tl.program_id(1)
    h_block = tl.program_id(2)

    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Concatenation rule: if m < T -> encoder[b, m, h]; else -> hidden[b, m - T, h]
    is_encoder = m < T

    if is_encoder:
        src_ptr = encoder_ptr + b * stride_e_b + m * stride_e_t + h_offsets * stride_e_h
    else:
        src_idx = m - T
        src_ptr = hidden_ptr + b * stride_h_b + src_idx * stride_h_i + h_offsets * stride_h_h

    val = tl.load(src_ptr, mask=mask_h, other=0.0)
    out_ptr_m = out_ptr + b * stride_o_b + m * stride_o_l + h_offsets * stride_o_h
    tl.store(out_ptr_m, val, mask=mask_h)


@triton.jit
def batched_matmul_2d_kernel(
    A_ptr, WT_ptr, C_ptr,
    B, M, N, K,
    stride_a_b, stride_a_m, stride_a_k,
    stride_w_k, stride_w_n,  # WT shape [K, N] (process_weight.T)
    stride_c_b, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A_tile: [BLOCK_M, BLOCK_K] from A[b, m, k]
        a_ptrs = A_ptr + b * stride_a_b + m_offsets[:, None] * stride_a_m + k_offsets[None, :] * stride_a_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load WT_tile: [BLOCK_K, BLOCK_N] from WT[k, n] which has shape [K, N]
        wt_ptrs = WT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        wt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        wt_tile = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        acc += tl.dot(a_tile.to(tl.float32), wt_tile.to(tl.float32))

    # Store C[b, m, n]
    c_ptrs = C_ptr + b * stride_c_b + m_offsets[:, None] * stride_c_m + n_offsets[None, :] * stride_c_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Triton kernel performs concatenation along sequence dimension.
        - Triton kernel performs batched matmul: processed = concatenated @ process_weight.T.
        - Split back into processed_encoder and processed_hidden.
        """
        # Ensure contiguous inputs
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        B, T, H = encoder.shape
        I = hidden.shape[1]
        L = T + I

        # 1) Concatenate along sequence dimension using Triton: out_cat [B, L, H]
        out_cat = torch.empty((B, L, H), device=encoder.device, dtype=encoder.dtype)

        BLOCK_H = 128  # tile along H for concatenation
        grid_cat = (B, L, triton.cdiv(H, BLOCK_H))
        cat_seq_kernel[grid_cat](
            encoder, hidden, out_cat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=1, num_stages=1,
        )

        # 2) Linear projection using Triton GEMM: processed = out_cat @ process_weight.T
        WT = weight.transpose(0, 1).contiguous()  # [H, H]
        processed = torch.empty((B, L, H), device=out_cat.device, dtype=out_cat.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_2d_kernel[grid_gemm](
            out_cat, WT, processed,
            B, L, H, H,  # M=L, N=H, K=H
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            WT.stride(0), WT.stride(1),  # WT strides (k, n)
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
