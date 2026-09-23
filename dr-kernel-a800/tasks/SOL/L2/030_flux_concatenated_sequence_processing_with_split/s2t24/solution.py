import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_l, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, L, ceil(H / BLOCK_H))
    b = tl.program_id(0)
    l = tl.program_id(1)
    tile_k = tl.program_id(2)

    k = tile_k * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_k = k < H

    # Determine source: first T rows from encoder, next I rows from hidden
    is_encoder = l < T

    if is_encoder:
        src = encoder_ptr + b * stride_e_b + l * stride_e_t + k * stride_e_h
    else:
        m = l - T
        src = hidden_ptr + b * stride_h_b + m * stride_h_i + k * stride_h_h

    out = out_ptr + b * stride_o_b + l * stride_o_l + k * stride_o_h

    vals = tl.load(src, mask=mask_k, other=0.0)
    tl.store(out, vals, mask=mask_k)


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    B, M, N, K,
    stride_A_b, stride_A_m, stride_A_k,
    stride_B_k, stride_B_n,
    stride_C_b, stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A[b, m, k]
        A_ptrs = A_ptr + b * stride_A_b + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_k
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B[k, n] = process_weight.T
        B_ptrs = B_ptr + k_offsets[:, None] * stride_B_k + n_offsets[None, :] * stride_B_n
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        bmat = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Fused multiply-add
        acc += tl.dot(a, bmat)

    C_ptrs = C_ptr + b * stride_C_b + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
          1) Concatenate encoder_hidden_states and hidden_states along sequence dimension
          2) Apply linear projection to the concatenated sequence
          3) Split back into separate encoder and image streams

        Returns:
          processed_encoder: [B, T, H]
          processed_hidden: [B, I, H]
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, dim, H]"
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "Batch and hidden_dim must match across inputs"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # 1) Concatenate into out_cat [B, L, H], L = T + I
        L = T + I
        out_cat = torch.empty((B, L, H), device=encoder.device, dtype=encoder.dtype)

        # Launch Triton concat kernel: grid (B, L, ceil(H / BLOCK_H))
        BLOCK_H = 64
        grid = (B, L, triton.cdiv(H, BLOCK_H))
        concat_seq_kernel[grid](
            encoder, hidden, out_cat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection: C = out_cat @ process_weight.T  -> [B, L, H]
        WT = weight.t().contiguous()  # [H, H]
        C = torch.empty((B, L, H), device=out_cat.device, dtype=out_cat.dtype)

        # Triton batched matmul kernel: grid (B, ceil(L/BLOCK_M), ceil(H/BLOCK_N))
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            out_cat, WT, C,
            B, L, H, H,  # M=L, N=H, K=H
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            WT.stride(0), WT.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into separate streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
