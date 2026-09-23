import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder and hidden along sequence dimension.
# Inputs:
#   encoder: [B, T, H], out_cat: [B, L, H]
# For each (b, m, k): if m < T -> out_cat[b, m, k] = encoder[b, m, k]; else -> out_cat[b, m, k] = hidden[b, m-T, k]
@triton.jit
def concat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    # strides
    enc_bs, enc_ts, enc_hs,
    hid_bs, hid_is, hid_hs,
    out_bs, out_ls, out_hs,
):
    b = tl.program_id(0)
    m = tl.program_id(1)  # sequence position in out_cat
    k = tl.program_id(2)  # hidden dim index

    # bounds check (grid ensures in-bounds, but keep it defensive)
    if (b >= 0 and b < B) and (m >= 0 and m < T + I) and (k >= 0 and k < H):
        if m < T:
            # read from encoder
            val = tl.load(encoder_ptr + b * enc_bs + m * enc_ts + k * enc_hs)
            tl.store(out_ptr + b * out_bs + m * out_ls + k * out_hs, val)
        else:
            # read from hidden at offset (m - T)
            val = tl.load(hidden_ptr + b * hid_bs + (m - T) * hid_is + k * hid_hs)
            tl.store(out_ptr + b * out_bs + m * out_ls + k * out_hs, val)


# Triton kernel: batched GEMM for C[b, m, n] = sum_k A[b, m, k] * W_T[k, n]
# A is the concatenated sequence out_cat of shape [B, L, H]
# W_T is process_weight.T of shape [H, H]
# Output C is [B, L, H]
@triton.jit
def batched_matmul_kernel(
    A_ptr, WT_ptr, C_ptr,
    B, M_total, N_total, K_total,
    # strides for A
    A_bs, A_ms, A_ks,
    # strides for WT (matrix)
    WT_ks, WT_ns,
    # strides for C
    C_bs, C_ms, C_ns,
    # tiling
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # masks for output tiles
    m_mask = m_offsets < M_total
    n_mask = n_offsets < N_total

    # accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension in blocks
    for k0 in range(0, K_total, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K_total

        # load A tiles: shape [BLOCK_M, BLOCK_K]
        # A[b, m, k] -> pointer arithmetic using strides
        # For all m in m_offsets and k in k_offsets
        a_ptrs = A_ptr + pid_b * A_bs + m_offsets[:, None] * A_ms + k_offsets[None, :] * A_ks
        a_mask = (m_mask[:, None]) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # cast to fp32 for accumulation
        a = a.to(tl.float32)

        # load W_T tiles: shape [BLOCK_K, BLOCK_N]
        # WT[k, n] -> strides WT_ks, WT_ns
        wt_ptrs = WT_ptr + k_offsets[:, None] * WT_ks + n_offsets[None, :] * WT_ns
        wt_mask = (k_mask[:, None]) & (n_mask[None, :])
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)
        wt = wt.to(tl.float32)

        # accumulate: acc += a @ wt
        acc += tl.dot(a, wt)

    # store results back to C[b, m, n] with masks
    c_ptrs = C_ptr + pid_b * C_bs + m_offsets[:, None] * C_ms + n_offsets[None, :] * C_ns
    c_mask = (m_mask[:, None]) & (n_mask[None, :])
    # acc is fp32; Triton will cast to C dtype on store if needed
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward that:
          - Concatenates encoder_hidden_states and hidden_states along the sequence dimension in Triton.
          - Performs the linear projection (matmul with process_weight.T) in Triton.
          - Splits the result back into processed_encoder and processed_hidden.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, S, H]"
        assert process_weight.dim() == 2, "process_weight must be [H, H]"
        B = hidden_states.size(0)
        T = encoder_hidden_states.size(1)
        I = hidden_states.size(1)
        H = hidden_states.size(2)
        L = T + I

        # Prepare inputs: ensure contiguous and on CUDA
        # We will not change dtype; keep as-is for consistency with PyTorch baseline.
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()  # [H, H]
        device = encoder.device

        # 1) Concatenate along sequence dimension using Triton
        out_cat = torch.empty((B, L, H), device=device, dtype=encoder.dtype)

        # Launch grid over (B, L, H)
        grid = (B, L, H)
        concat_seq_kernel[grid](
            encoder, hidden, out_cat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Linear projection: out_cat @ process_weight.T using Triton GEMM
        WT = weight.t().contiguous()  # [H, H]
        processed = torch.empty((B, L, H), device=device, dtype=encoder.dtype)

        # GEMM launch parameters
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_gemm = (
            B,
            triton.cdiv(L, BLOCK_M),
            triton.cdiv(H, BLOCK_N),
        )

        batched_matmul_kernel[grid_gemm](
            out_cat, WT, processed,
            B, L, H, H,  # M_total = L, N_total = H, K_total = H
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),  # A strides
            WT.stride(0), WT.stride(1),  # WT is [H, H], strides
            processed.stride(0), processed.stride(1), processed.stride(2),  # C strides
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
