import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    encoder_ptr, hidden_ptr, cat_ptr,
    B, T, P, K,
    encoder_stride_b, encoder_stride_l, encoder_stride_k,
    hidden_stride_b, hidden_stride_l, hidden_stride_k,
    cat_stride_b, cat_stride_l, cat_stride_k,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(T+P, BLOCK_L), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    l_block = tl.program_id(1)
    k_block = tl.program_id(2)

    l = l_block * BLOCK_L + tl.arange(0, BLOCK_L)  # sequence positions [0..T+P)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden_dim indices

    # mask for valid l and k
    mask_l = l < (T + P)
    mask_k = k < K

    # decide source: if l < T -> encoder, else -> hidden
    use_encoder = l < T

    # Compute pointers for loads
    # encoder[b, l, k]
    # Note: for l where use_encoder is False, this pointer won't be used due to masked load.
    enc_ptrs = encoder_ptr + b * encoder_stride_b + l[:, None] * encoder_stride_l + k[None, :] * encoder_stride_k
    # hidden[b, l - T, k]
    hid_ptrs = hidden_ptr + b * hidden_stride_b + (l[:, None] - T) * hidden_stride_l + k[None, :] * hidden_stride_k

    # Masks for loads
    mask_load = mask_l[:, None] & mask_k[None, :] & use_encoder[:, None]
    mask_load_hidden = mask_l[:, None] & mask_k[None, :] & (~use_encoder)[:, None]

    # Load from appropriate source
    enc_vals = tl.load(enc_ptrs, mask=mask_load, other=0.0)
    hid_vals = tl.load(hid_ptrs, mask=mask_load_hidden, other=0.0)
    # Combine: where use_encoder, enc_vals, else hid_vals
    vals = tl.where(use_encoder[:, None] & mask_l[:, None], enc_vals, 0.0) + tl.where((~use_encoder)[:, None] & mask_l[:, None], hid_vals, 0.0)

    # Store to cat[b, l, k]
    cat_ptrs = cat_ptr + b * cat_stride_b + l[:, None] * cat_stride_l + k[None, :] * cat_stride_k
    store_mask = mask_l[:, None] & mask_k[None, :]
    tl.store(cat_ptrs, vals, mask=store_mask)


@triton.jit
def _matmul_btl_kernel(
    A_ptr, W_ptr, C_ptr,
    B, T, P, K,
    A_stride_b, A_stride_l, A_stride_k,
    W_stride_k_in, W_stride_k_out,  # W is [K, K], strides for reading rows and writing cols
    C_stride_b, C_stride_l, C_stride_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    # Here M = B*(T+P), N = K
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Compute tile indices
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in A, i.e., positions in concatenated sequence
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # output hidden_dim indices

    # Map m_offsets to (batch b, l) for A[b, l, :]
    # Since we launch grid over b first, all m_offsets for this program correspond to the same batch b.
    total_L = T + P
    # We need to iterate over l per m: m = b*total_L + l -> l = m - b*total_L
    # We will do a reduction loop over K with BLOCK_K chunks.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)

        # Build A pointers for this tile: A[b, l, k_offsets]
        # l per m: l = m_offsets - b*total_L
        l_vals = m_offsets - b * total_L
        valid_m = m_offsets < (B * total_L)
        valid_l = (l_vals >= 0) & (l_vals < total_L) & valid_m
        A_ptrs = A_ptr + b * A_stride_b + l_vals[:, None] * A_stride_l + k_offsets[None, :] * A_stride_k
        A_mask = valid_l[:, None] & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Build W pointers for this tile: W[k_offsets, n_offsets]
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride_k_in + n_offsets[None, :] * W_stride_k_out
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < K)
        w = tl.load(W_ptrs, mask=W_mask, other=0.0)

        acc += tl.dot(a, w)

    # Store results to C[b, m_offsets, n_offsets]
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_l + n_offsets[None, :] * C_stride_k
    store_mask = (m_offsets[:, None] < (B * (T + P))) & (n_offsets[None, :] < K)
    tl.store(C_ptrs, acc, mask=store_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_B: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(B, BLOCK_B), cdiv(T, BLOCK_T), cdiv(K, BLOCK_K))
    b_block = tl.program_id(0)
    t_block = tl.program_id(1)
    k_block = tl.program_id(2)

    b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
    t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_b = b < B
    mask_t = t < T
    mask_k = k < K

    # Pointers for C[:, :T, :]
    C_ptrs = C_ptr + b[:, None, None] * C_stride_b + t[None, :, None] * C_stride_l + k[None, None, :] * C_stride_k
    mask = mask_b[:, None, None] & mask_t[None, :, None] & mask_k[None, None, :]
    vals = tl.load(C_ptrs, mask=mask, other=0.0)

    # Store to out[b, t, k]
    out_ptrs = out_ptr + b[:, None, None] * out_stride_b + t[None, :, None] * out_stride_l + k[None, None, :] * out_stride_k
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_B: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(B, BLOCK_B), cdiv(P, BLOCK_P), cdiv(K, BLOCK_K))
    b_block = tl.program_id(0)
    p_block = tl.program_id(1)
    k_block = tl.program_id(2)

    b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
    p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_b = b < B
    mask_p = p < P
    mask_k = k < K

    # Copy C[:, T:, :] -> out[:, p, :]
    # l indices for source: l = T + p
    total_L = T + P
    C_ptrs = C_ptr + b[:, None, None] * C_stride_b + (T + p[None, :]) * C_stride_l + k[None, None, :] * C_stride_k
    mask = mask_b[:, None, None] & mask_p[None, :, None] & mask_k[None, None, :]
    vals = tl.load(C_ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + b[:, None, None] * out_stride_b + p[None, :, None] * out_stride_l + k[None, None, :] * out_stride_k
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate (encoder, image) via Triton kernel
          - Linear projection via Triton matmul kernel
          - Split into encoder and image outputs via Triton kernels
        Returns (processed_encoder: [B, T, K], processed_hidden: [B, P, K])
        """
        B, P, K = hidden_states.shape
        _, T, _ = encoder_hidden_states.shape
        assert process_weight.shape == (K, K), "process_weight must be [K, K]"

        # 1) Allocate concatenated tensor Acat [B, T+P, K] in float32
        total_L = T + P
        Acat = torch.empty((B, total_L, K), device=hidden_states.device, dtype=torch.float32)

        # Strides
        eb, et, ek = encoder_hidden_states.stride()
        hb, hp, hk = hidden_states.stride()
        cab, cal, cak = Acat.stride()

        # Launch concatenation kernel
        BLOCK_L = 64
        BLOCK_K = 64
        grid_cat = (B, triton.cdiv(total_L, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concatenate_sequences_kernel[grid_cat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            eb, et, ek,
            hb, hp, hk,
            cab, cal, cak,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Triton GEMM: Acat [B*(T+P), K] @ process_weight.T [K, K] -> C [B*(T+P), K]
        # View Acat as [M, K], W as [K, K]
        A = Acat  # [B, total_L, K], contiguous by construction
        W = process_weight.t().contiguous()  # [K, K], contiguous for efficient loads

        # Prepare output flat [M, K]
        M = B * total_L
        C = torch.empty((M, K), device=hidden_states.device, dtype=torch.float32)

        # Strides for A (as [M, K])
        # A is [B, total_L, K]; we index A by m = b*total_L + l, so per (b,l) slice:
        A_stride_b = total_L * K
        A_stride_l = K
        A_stride_k = 1

        # For W [K, K]: strides (row-major)
        W_stride_k_in = 1  # along k_in (rows)
        W_stride_k_out = K  # along k_out (cols)

        # For C [M, K]:
        C_stride_m = K
        C_stride_k = 1
        # Note: C has no "l" dimension; we flatten to [M, K], so C_stride_l is not used.

        # Launch GEMM kernel
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_btl_kernel[grid_gemm](
            A, W, C,
            B, T, P, K,
            A_stride_b, A_stride_l, A_stride_k,
            W_stride_k_in, W_stride_k_out,
            C_stride_m, 0, C_stride_k,  # C[l dimension is implicit in M; second stride not used
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Reshape C to [B, total_L, K] and split via Triton
        C3 = C.view(B, total_L, K)

        # Processed encoder: [B, T, K]
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        BLOCK_Be = 64
        BLOCK_Te = 64
        BLOCK_Ke = 64
        grid_e = (triton.cdiv(B, BLOCK_Be), triton.cdiv(T, BLOCK_Te), triton.cdiv(K, BLOCK_Ke))
        _split_encoder_kernel[grid_e](
            C3, processed_encoder,
            B, T, K,
            C3.stride(0), C3.stride(1), C3.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_B=BLOCK_Be, BLOCK_T=BLOCK_Te, BLOCK_K=BLOCK_Ke,
            num_warps=4, num_stages=2
        )

        # Processed hidden: [B, P, K]
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)
        BLOCK_Bh = 64
        BLOCK_Ph = 64
        BLOCK_Kh = 64
        grid_h = (triton.cdiv(B, BLOCK_Bh), triton.cdiv(P, BLOCK_Ph), triton.cdiv(K, BLOCK_Kh))
        _split_hidden_kernel[grid_h](
            C3, processed_hidden,
            B, T, P, K,
            C3.stride(0), C3.stride(1), C3.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_B=BLOCK_Bh, BLOCK_P=BLOCK_Ph, BLOCK_K=BLOCK_Kh,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
