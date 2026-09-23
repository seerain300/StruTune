import torch
import triton
import triton.language as tl

@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr,  # *T, [B, T, H]
    hid_ptr,  # *T, [B, I, H]
    out_ptr,  # *T, [B, L, H], L = T + I
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along L
    BLOCK_K: tl.constexpr,  # tile along H
):
    # Grid: (B, ceil((T + I) / BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # along L = T + I
    k_offsets = tl.arange(0, BLOCK_K)                      # along H

    m_total = T + I
    mask_m = m_offsets < m_total
    mask_k = k_offsets < H

    # Loop over H in chunks to vectorize along K
    for k0 in range(0, H, BLOCK_K):
        k = k0 + k_offsets
        mask_k = k < H

        for m_i in range(0, BLOCK_M):
            m_idx = m_offsets[m_i]
            valid_m = m_idx < m_total

            # Select source: enc for m < T, hid for m >= T
            is_enc = valid_m & (m_idx < T)
            is_hid = valid_m & (m_idx >= T)

            # Compute pointers
            # enc[b, m, k] -> base over (b, k)
            enc_ptr_row = enc_ptr + b * enc_ptr.stride(0) + m_idx * enc_ptr.stride(1) + k * enc_ptr.stride(2)
            # hid[b, m - T, k] -> base over (b, k)
            hid_ptr_row = hid_ptr + b * hid_ptr.stride(0) + (m_idx - T) * hid_ptr.stride(1) + k * hid_ptr.stride(2)

            # Load with masks
            enc_val = tl.load(enc_ptr_row, mask=is_enc & mask_k, other=0.0)
            hid_val = tl.load(hid_ptr_row, mask=is_hid & mask_k, other=0.0)

            # Select appropriate value
            # If both enc and hid active, need to branch. Triton supports scalar if-else per lane.
            # We'll compute with a scalar if based on m_i.
            if is_enc:
                val = enc_val
            else:
                val = hid_val

            # Store into out[b, m, k]
            out_ptr_row = out_ptr + b * out_ptr.stride(0) + m_idx * out_ptr.stride(1) + k * out_ptr.stride(2)
            tl.store(out_ptr_row, val, mask=valid_m & mask_k)


@triton.jit
def _batched_matmul_2d_kernel_fp32(
    A_ptr,   # *float32, [B, M=L, K=H]
    WT_ptr,  # *float32, [P=H, Q=H] but used as B = (K=H) x N=H, so WT_ptr is [K, N] i.e., process_weight.T
    C_ptr,   # *float32, [B, M, N=H]
    B: tl.constexpr,  # batch size (unused but kept for clarity)
    M: tl.constexpr,  # seq length = T + I
    N: tl.constexpr,  # hidden dim H
    K: tl.constexpr,  # hidden dim H (reduction dim)
    strideA_b, strideA_m, strideA_k,
    strideWT_p, strideWT_q,   # WT is [K, N], so p=K, q=N
    strideC_b, strideC_m, strideC_n,
    BLOCK_M: tl.constexpr,    # tile over M
    BLOCK_N: tl.constexpr,    # tile over N
    BLOCK_K: tl.constexpr,    # reduction tile over K
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]
        mask_k = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * strideA_b + m_offsets[:, None] * strideA_m + k_offsets[None, :] * strideA_k
        a_mask = (m_offsets[:, None] < M) & (mask_k[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K], fp32

        # Load WT tile as [BLOCK_K, BLOCK_N]: WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * strideWT_p + n_offsets[None, :] * strideWT_q
        wt_mask = (mask_k[:, None]) & (n_offsets[None, :] < N)
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)  # [BLOCK_K, BLOCK_N], fp32

        # Accumulate
        acc += tl.dot(a, wt)  # [BLOCK_M, BLOCK_N]

    # Store C tile
    c_ptrs = C_ptr + b * strideC_b + m_offsets[:, None] * strideC_m + n_offsets[None, :] * strideC_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [B, I, H]
        # encoder_hidden_states: [B, T, H]
        # process_weight: [H, H] (no bias)
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        device = hidden_states.device

        # 1) Concatenate into [B, L, H] using Triton
        out_cat = torch.empty((B, L, H), device=device, dtype=hidden_states.dtype)
        # Make inputs contiguous for predictable strides
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        out = out_cat.contiguous()

        # Choose tiling for concat (simple and robust)
        BLOCK_M = 128  # tile along L
        BLOCK_K = 64   # tile along H
        grid_concat = (B, triton.cdiv(L, BLOCK_M))
        _concatenate_seqs_kernel[grid_concat](
            enc, hid, out,
            B=B, T=T, I=I, H=H,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection: out_cat @ process_weight.T, computed by Triton in fp32
        WT = process_weight.t().contiguous().to(torch.float32)  # [H, H]
        A = out  # [B, L, H], using out_cat as A for performance
        # Allocate output C in fp32
        C = torch.empty((B, L, H), device=device, dtype=torch.float32)
        strideA_b, strideA_m, strideA_k = A.stride()
        strideWT_p, strideWT_q = WT.stride()  # WT is [H, H], so p=H (K), q=H (N)
        strideC_b, strideC_m, strideC_n = C.stride()

        # 2D tiling over (M=L, N=H), reduction over K=H
        BLOCK_M_mm = 64
        BLOCK_N_mm = 64
        BLOCK_K_mm = 32
        grid_mm = (B, triton.cdiv(L, BLOCK_M_mm), triton.cdiv(H, BLOCK_N_mm))
        _batched_matmul_2d_kernel_fp32[grid_mm](
            A, WT, C,
            B=B, M=L, N=H, K=H,
            strideA_b=strideA_b, strideA_m=strideA_m, strideA_k=strideA_k,
            strideWT_p=strideWT_p, strideWT_q=strideWT_q,
            strideC_b=strideC_b, strideC_m=strideC_m, strideC_n=strideC_n,
            BLOCK_M=BLOCK_M_mm, BLOCK_N=BLOCK_N_mm, BLOCK_K=BLOCK_K_mm,
            num_warps=4, num_stages=3,
        )

        # 3) Split results back into encoder and hidden streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
