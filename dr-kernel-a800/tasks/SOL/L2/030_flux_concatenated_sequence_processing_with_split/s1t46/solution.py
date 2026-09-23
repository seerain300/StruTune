import torch
import triton
import triton.language as tl


@triton.jit
def _concat_encoder_hidden_kernel(
    encoder_ptr,  # *float, [B, T, K]
    hidden_ptr,   # *float, [B, P, K]
    out_ptr,      # *float, [B, T+P, K]
    B, T, P, K,
    BLOCK_K: tl.constexpr,
):
    # program ids: batch, sequence position, K-tile
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Determine source tensor
    is_encoder = l < T
    # Compute source offsets (assuming contiguous [B, L, K])
    if is_encoder:
        src_b = b
        src_l = l
        base_src = src_b * (T * K) + src_l * K
    else:
        src_b = b
        src_l = l - T
        base_src = src_b * (P * K) + src_l * K

    # Output base (assuming contiguous [B, T+P, K])
    base_out = b * ((T + P) * K) + l * K

    vals = tl.load(encoder_ptr + base_src + k_offsets, mask=k_mask, other=0.0)
    # If l >= T, the loaded vals are from encoder. For hidden, we would load from hidden_ptr + base_src + k_offsets; but we can avoid dual loads by checking and selecting:
    # Better approach: load from correct source pointer based on is_encoder
    if is_encoder:
        # already loaded from encoder
        pass
    else:
        vals = tl.load(hidden_ptr + base_src + k_offsets, mask=k_mask, other=0.0)

    tl.store(out_ptr + base_out + k_offsets, vals, mask=k_mask)


@triton.jit
def _gemm_matmul_kernel(
    A_ptr,  # *float, [M, K] where M = B*(T+P), contiguous row-major
    W_ptr,  # *float, [K, K], contiguous row-major
    C_ptr,  # *float, [M, K], contiguous row-major
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (b, m_tile, n_tile). Triton allows program_id(0) as batch index.
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K in chunks
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: A[i, j] offset = i*K + j
        A_ptrs = A_ptr + m_offsets[:, None] * K + k_offsets[None, :]
        A_mask = (m_offsets[:, None] < M) & (k_mask[None, :])
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W tile: W[i, j] offset = i*K + j
        W_ptrs = W_ptr + k_offsets[:, None] * K + n_offsets[None, :]
        W_mask = (k_mask[:, None]) & (n_offsets[None, :] < N)
        W_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store C: C[i, j] offset = i*N + j
    C_ptrs = C_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    M_mask = m_offsets[:, None] < M
    N_mask = n_offsets[None, :] < N
    store_mask = M_mask & N_mask
    tl.store(C_ptrs, acc, mask=store_mask)


@triton.jit
def _split_encoder_kernel(
    C_flat_ptr,   # *float, [M, K] where M = B*(T+P)
    out_ptr,      # *float, [B, T, K], contiguous
    B, T, P, K,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T, ceil_div(K, BLOCK_K))
    b = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # M index for this (b, t): M = b*(T+P) + t
    M_idx = b * (T + P) + t
    out_base = (b * T + t) * K

    vals = tl.load(C_flat_ptr + M_idx * K + k_offsets, mask=k_mask, other=0.0)
    tl.store(out_ptr + out_base + k_offsets, vals, mask=k_mask)


@triton.jit
def _split_hidden_kernel(
    C_flat_ptr,   # *float, [M, K] where M = B*(T+P)
    out_ptr,      # *float, [B, P, K], contiguous
    B, T, P, K,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, P, ceil_div(K, BLOCK_K))
    b = tl.program_id(0)
    p = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # M index for this (b, p): M = b*(T+P) + (p + T)
    M_idx = b * (T + P) + (p + T)
    out_base = (b * P + p) * K

    vals = tl.load(C_flat_ptr + M_idx * K + k_offsets, mask=k_mask, other=0.0)
    tl.store(out_ptr + out_base + k_offsets, vals, mask=k_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B = hidden_states.shape[0]
        P = hidden_states.shape[1]  # img_seq_len
        T = encoder_hidden_states.shape[1]  # text_seq_len
        K = hidden_states.shape[2]  # hidden_dim

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure contiguous inputs
        encoder = encoder_hidden_states.contiguous()      # [B, T, K]
        hidden = hidden_states.contiguous()              # [B, P, K]
        weight_t = process_weight.transpose(0, 1).contiguous()  # [K, K]

        total_L = T + P

        # Step 1: Triton concatenation -> Acat [B, T+P, K]
        Acat = torch.empty((B, total_L, K), device=device, dtype=dtype)

        BLOCK_K = 64
        grid_concat = (B, total_L, triton.cdiv(K, BLOCK_K))
        _concat_encoder_hidden_kernel[grid_concat](
            encoder, hidden, Acat,
            B, T, P, K,
            BLOCK_K=BLOCK_K, num_warps=4, num_stages=2
        )

        # Step 2: GEMM via Triton: Acat [B, T+P, K] flattened to [M, K], W [K, K]
        M = B * total_L
        C_flat = torch.empty((M, K), device=device, dtype=dtype)

        # Create A_2d [M, K] contiguous from Acat
        A_2d = Acat.reshape(M, K).contiguous()

        # GEMM kernel config
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_Kr = 32

        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _gemm_matmul_kernel[grid_gemm](
            A_2d, weight_t, C_flat,
            M, K, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_Kr,
            num_warps=4, num_stages=3
        )

        # Step 3: Triton splitting into encoder and hidden streams
        processed_encoder = torch.empty((B, T, K), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=dtype)

        BLOCK_K_s = 64
        grid_e = (B, T, triton.cdiv(K, BLOCK_K_s))
        _split_encoder_kernel[grid_e](
            C_flat, processed_encoder,
            B, T, P, K,
            BLOCK_K=BLOCK_K_s, num_warps=4, num_stages=2
        )

        grid_h = (B, P, triton.cdiv(K, BLOCK_K_s))
        _split_hidden_kernel[grid_h](
            C_flat, processed_hidden,
            B, T, P, K,
            BLOCK_K=BLOCK_K_s, num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
