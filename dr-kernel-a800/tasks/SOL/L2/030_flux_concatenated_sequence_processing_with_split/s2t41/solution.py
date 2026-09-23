import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr,              # *T, [B, T, H]
    hid_ptr,              # *T, [B, I, H]
    out_ptr,              # *T, [B, L, H], L = T + I
    B, T, I, H,           # int32
    BLOCK_M: tl.constexpr,  # tile along L
    BLOCK_K: tl.constexpr,  # tile along H
):
    # Grid: (B, ceil(L / BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    L = T + I
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = tl.arange(0, BLOCK_K)                      # [BLOCK_K]

    # Loop over H in chunks
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Iterate over each m index in the tile
        for m_idx in range(BLOCK_M):
            m = m_offsets[m_idx]
            valid_m = m < L

            # Compute output pointer: out[b, m, k]
            out_row_ptr = out_ptr + b * out_ptr.stride(0) + m * out_ptr.stride(1) + k_offsets * out_ptr.stride(2)

            # If m < T: use encoder; else: use hidden at (m - T)
            enc_row_ptr = enc_ptr + b * enc_ptr.stride(0) + m * enc_ptr.stride(1) + k_offsets * enc_ptr.stride(2)
            hid_row_ptr = hid_ptr + b * hid_ptr.stride(0) + (m - T) * hid_ptr.stride(1) + k_offsets * hid_ptr.stride(2)

            enc_valid = valid_m & (m < T)
            hid_valid = valid_m & (m >= T)

            enc_val = tl.load(enc_row_ptr, mask=enc_valid & mask_k, other=0.0)
            hid_val = tl.load(hid_row_ptr, mask=hid_valid & mask_k, other=0.0)

            out_val = tl.where(m < T, enc_val, hid_val)
            tl.store(out_row_ptr, out_val, mask=valid_m & mask_k)


@triton.jit
def _batched_matmul_kernel_fp32(
    A_ptr,       # *float32, [B, M, K], A = concatenated [B, L, H]
    WT_ptr,      # *float32, [K, N] = process_weight.T [H, H]
    C_ptr,       # *float32, [B, M, N] = output [B, L, H]
    B, M, N, K,  # int32 sizes
    strideA_b, strideA_m, strideA_k,
    strideWT_p, strideWT_q,
    strideC_b, strideC_m, strideC_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)   # along M (L)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)   # along N (H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (H)
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[b, m, k] -> (BLOCK_M, BLOCK_K)
        A_tile_ptr = A_ptr + b * strideA_b + m_offsets[:, None] * strideA_m + k_offsets[None, :] * strideA_k
        A_mask = (m_offsets[:, None] < M) & (mask_k[None, :])
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)  # fp32

        # Load WT[k, n] -> (BLOCK_K, BLOCK_N)
        WT_tile_ptr = WT_ptr + k_offsets[:, None] * strideWT_p + n_offsets[None, :] * strideWT_q
        WT_mask = (mask_k[:, None]) & (n_offsets[None, :] < N)
        WT_tile = tl.load(WT_tile_ptr, mask=WT_mask, other=0.0)  # fp32

        acc += tl.dot(A_tile, WT_tile)

    # Store C[b, m, n] = acc
    C_tile_ptr = C_ptr + b * strideC_b + m_offsets[:, None] * strideC_m + n_offsets[None, :] * strideC_n
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "inputs must be [B, dim, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and hidden_states.shape[0] == B, "batch size must match"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        device = hidden_states.device

        # 1) Triton concatenation: out_cat [B, L, H], L = T + I
        L = T + I
        out_cat = torch.empty((B, L, H), device=device, dtype=hidden_states.dtype)

        enc = encoder_hidden_states.contiguous()   # [B, T, H]
        hid = hidden_states.contiguous()           # [B, I, H]
        out_cat = out_cat.contiguous()             # [B, L, H]

        # Launch concatenation kernel
        BLOCK_M = 128
        BLOCK_K = 64
        grid_cat = (B, triton.cdiv(L, BLOCK_M))
        _concatenate_seqs_kernel[grid_cat](
            enc, hid, out_cat,
            B, T, I, H,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Triton batched GEMM: C = out_cat @ process_weight.T (fp32 accumulation)
        WT = process_weight.t().contiguous().to(torch.float32)  # [H, H]
        A = out_cat                                     # [B, L, H]
        C = torch.empty((B, L, H), device=device, dtype=torch.float32)  # output [B, L, H]

        strideA_b, strideA_m, strideA_k = A.stride()
        strideWT_p, strideWT_q = WT.stride()  # WT is [H, H]
        strideC_b, strideC_m, strideC_n = C.stride()

        # Tiling params
        BLOCK_M_mm = 64
        BLOCK_N_mm = 64
        BLOCK_K_mm = 32
        grid_mm = (B, triton.cdiv(L, BLOCK_M_mm), triton.cdiv(H, BLOCK_N_mm))

        _batched_matmul_kernel_fp32[grid_mm](
            A, WT, C,
            B, L, H, H,
            strideA_b, strideA_m, strideA_k,
            strideWT_p, strideWT_q,
            strideC_b, strideC_m, strideC_n,
            BLOCK_M=BLOCK_M_mm, BLOCK_N=BLOCK_N_mm, BLOCK_K=BLOCK_K_mm,
            num_warps=4, num_stages=3,
        )

        # 3) Split back: C has shape [B, L, H]
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
