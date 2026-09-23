import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    encoder_ptr,      # *fp32, [B, T, K]
    hidden_ptr,       # *fp32, [B, P, K]
    out_ptr,          # *fp32, [B, T+P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_L: tl.constexpr,  # along T+P
    BLOCK_K: tl.constexpr,  # along K
):
    # Grid: (B, cdiv(T+P, BLOCK_L), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    l_block = tl.program_id(1)
    k_block = tl.program_id(2)

    L = T + P

    l_offsets = l_block * BLOCK_L + tl.arange(0, BLOCK_L)  # positions within concatenated sequence
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # columns within K

    mask_l = l_offsets < L
    mask_k = k_offsets < K

    # For l < T: source is encoder[b, l, :], else: hidden[b, l - T, :]
    is_encoder = l_offsets < T  # boolean mask per l

    # Compute source row index for encoder/hidden
    src_row = tl.where(is_encoder, l_offsets, l_offsets - T)  # int32 vector

    # Compute source and destination pointers
    # encoder: [B, T, K] -> row offset = b*T + l, col = k
    encoder_row_ptr = encoder_ptr + b * T + src_row * K
    # hidden: [B, P, K] -> row offset = b*P + (l - T), col = k
    hidden_row_ptr = hidden_ptr + b * P + (src_row - T) * K

    # Select source pointer based on is_encoder
    # Triton doesn't support dynamic pointer selection; we compute both and then combine via where
    # Build pointer matrices by broadcasting k_offsets
    # Load encoder tile
    enc_ptrs = encoder_row_ptr[:, None] + k_offsets[None, :]
    enc_vals = tl.load(enc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)

    # Load hidden tile
    hid_ptrs = hidden_row_ptr[:, None] + k_offsets[None, :]
    hid_vals = tl.load(hid_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)

    # Choose which to write: if is_encoder True -> enc_vals else hid_vals
    # Use tl.where on pointers isn't possible; instead compute a selection matrix:
    select = is_encoder[:, None]  # [BLOCK_L, 1] broadcast to [BLOCK_L, BLOCK_K]
    vals = tl.where(select, enc_vals, hid_vals)

    # Destination out_ptr: [B, T+P, K] -> row = l_offsets, col = k_offsets
    out_row_base = out_ptr + b * (L * K)
    out_ptrs = out_row_base + l_offsets[:, None] * K + k_offsets[None, :]
    store_mask = mask_l[:, None] & mask_k[None, :]
    tl.store(out_ptrs, vals, mask=store_mask)


@triton.jit
def _matmul_kernel(
    A_ptr,       # *fp32, [M, K], M = B*(T+P)
    W_ptr,       # *fp32, [K, K]
    C_ptr,       # *fp32, [M, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    M: tl.constexpr,  # M = B*(T+P)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # equals K
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(M, BLOCK_M), cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N], typically [0, K)

    mask_m = m_offsets < M
    mask_n = n_offsets < K

    # Map m_offsets to batch and position within concatenated sequence
    L = T + P
    b_idx = b
    pos = m_offsets % L  # position within [0, L)
    batch_row_in_A = m_offsets // L  # already b, but pos is sufficient since we don't use batch index in A rows
    # We don't need batch index in A since A is flattened by row. Using pos and b_idx is sufficient.

    # For each m, its row in A is m_offsets. We load A rows directly by m_offsets.
    A_row_ptr = A_ptr + m_offsets * K  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_row_ptr[:, None] + k_offsets_k[None, :]
        A_vals = tl.load(A_tile_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N], W is [K, K]
        W_tile_ptr = W_ptr + k_offsets_k[:, None] * K + n_offsets[None, :]
        W_vals = tl.load(W_tile_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(A_vals, W_vals)

    # Store C tile: C is [M, K], row index m_offsets, col n_offsets
    C_row_base = C_ptr + b_idx * (M * K)
    C_ptrs = C_row_base + m_offsets[:, None] * K + n_offsets[None, :]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=store_mask)


@triton.jit
def _split_encoder_kernel(
    C_flat_ptr,       # *fp32, [B, T, K], viewed as [M, K] where M = B*T
    out_encoder_ptr,  # *fp32, [B, T, K]
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(T, BLOCK_T), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    k_block = tl.program_id(2)

    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_t = t_offsets < T
    mask_k = k_offsets < K

    # M = B*T
    M = B * T

    # Source row in C_flat: m = b*T + t
    src_m = b * T + t_offsets
    src_ptrs = C_flat_ptr + src_m[:, None] * K + k_offsets[None, :]
    vals = tl.load(src_ptrs, mask=mask_t[:, None] & mask_k[None, :], other=0.0)

    # Destination: out_encoder_ptr at [b, t_offsets, k_offsets]
    out_base = out_encoder_ptr + b * (T * K)
    dest_ptrs = out_base + t_offsets[:, None] * K + k_offsets[None, :]
    tl.store(dest_ptrs, vals, mask=mask_t[:, None] & mask_k[None, :])


@triton.jit
def _split_hidden_kernel(
    C_flat_ptr,       # *fp32, [B, T+P, K], viewed as [M, K] where M = B*(T+P)
    out_hidden_ptr,   # *fp32, [B, P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(P, BLOCK_P), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    p_block = tl.program_id(1)
    k_block = tl.program_id(2)

    p_offsets = p_block * BLOCK_P + tl.arange(0, BLOCK_P)  # [BLOCK_P]
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_p = p_offsets < P
    mask_k = k_offsets < K

    M = B * (T + P)

    # For each p in [0, P), source row is M + b*(T+P) + p, but since we view C_flat as [M, K], we need to map
    # The "hidden" part starts at row T in the concatenated sequence. So for out_hidden [B, P, K], its rows are
    # src_m = b*(T+P) + (p_offsets + T). Note: M does not include hidden rows because we are viewing C_flat as [M, K] where M = B*T for encoder.
    # We should instead access C_flat at rows M + b*(T+P) + p_offsets, but M here is for encoder. Instead, we access C_flat as
    # C_flat has M = B*(T+P). For hidden split, we need to read rows [B*(T+P)] + p_offsets starting from T. Since we passed C_flat for entire [B, T+P, K],
    # the correct source row is src_m = b*(T+P) + p_offsets.
    src_m = b * (T + P) + p_offsets
    src_ptrs = C_flat_ptr + src_m[:, None] * K + k_offsets[None, :]
    vals = tl.load(src_ptrs, mask=mask_p[:, None] & mask_k[None, :], other=0.0)

    # Destination out_hidden_ptr at [b, p_offsets, k_offsets]
    out_base = out_hidden_ptr + b * (P * K)
    dest_ptrs = out_base + p_offsets[:, None] * K + k_offsets[None, :]
    tl.store(dest_ptrs, vals, mask=mask_p[:, None] & mask_k[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and dtype
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"
        B, T, K = encoder_hidden_states.shape
        B2, P, K2 = hidden_states.shape
        assert B == B2 and K == K2, "Hidden and encoder shapes must match in batch and feature dimension"
        W = process_weight  # [K, K]
        assert W.shape[0] == K and W.shape[1] == K

        # 1) Triton concatenation: Acat [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=device, dtype=torch.float32)

        BLOCK_L = 64
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B=B, T=T, P=P, K=K,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Triton GEMM: C_flat [B*(T+P), K]
        M = B * L
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        # Use BLOCK_M as tile over M, BLOCK_N as tile over K, BLOCK_K for reduction
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_mm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_mm](
            Acat, W, C_flat,
            B=B, T=T, P=P, K=K, M=M,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Triton split into encoder and hidden
        # processed_encoder: [B, T, K]
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        BLOCK_T = 64
        BLOCK_K_split = 64
        grid_e = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_e](
            C_flat, processed_encoder,
            B=B, T=T, K=K,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        # processed_hidden: [B, P, K]
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)
        BLOCK_P = 64
        grid_h = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_h](
            C_flat, processed_hidden,
            B=B, T=T, P=P, K=K,
            BLOCK_P=BLOCK_P, BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
