import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dim,
# producing Acat [B, T+P, K] without torch.cat.
@triton.jit
def _concatenate_kernel(
    encoder_ptr,          # *fp32, [B, T, K]
    hidden_ptr,           # *fp32, [B, P, K]
    acat_ptr,             # *fp32, [B, T+P, K]
    B: tl.constexpr,
    T: tl.constexpr,
    P: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)  # batch
    l = tl.program_id(1)  # position in concatenated sequence [0, T+P)
    k_offsets = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source: first T rows from encoder, remaining from hidden
    is_encoder = l < T
    src_base = b * T * K + l * K + k_offsets
    dst_base = b * (T + P) * K + l * K + k_offsets

    if is_encoder:
        src_ptr = encoder_ptr + src_base
    else:
        src_ptr = hidden_ptr + b * P * K + (l - T) * K + k_offsets

    vals = tl.load(src_ptr, mask=mask_k, other=0.0)
    tl.store(acat_ptr + dst_base, vals, mask=mask_k)


# Triton kernel: GEMM on Acat [M, K] and W [K, K] -> C_flat [M, K]
# We use a 3D grid over (batch, row tiles, col tiles) to ensure robust tiling.
@triton.jit
def _matmul_kernel(
    A_ptr,       # *fp32, [M, K], M = B*(T+P)
    W_ptr,       # *fp32, [K, K]
    C_ptr,       # *fp32, [M, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    M: tl.constexpr,  # M = B*(T+P)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    m_offsets = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < K

    total = T + P  # rows per batch
    m_row = m_offsets // total  # which batch
    r = m_offsets % total       # which row within the concatenated sequence

    # Pointers for A rows (A is [M, K])
    A_row_ptr = A_ptr + m_offsets * K  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_row_ptr[:, None] + k_offsets[None, :]
        A_vals = tl.load(A_tile_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # W tile: [BLOCK_K, BLOCK_N], W is [K, K]
        W_tile_ptr = W_ptr + k_offsets[:, None] * K + n_offsets[None, :]
        W_vals = tl.load(W_tile_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(A_vals, W_vals)

    # Store C tile: C is [M, K]
    C_tile_ptr = C_ptr + m_offsets[:, None] * K + n_offsets[None, :]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_tile_ptr, acc, mask=store_mask)


# Triton kernel: split C [B, T+P, K] into encoder and hidden parts
@triton.jit
def _split_encoder_kernel(
    C_ptr,          # *fp32, [B, T+P, K]
    out_ptr,        # *fp32, [B, T, K]
    B: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)  # batch
    d = tl.program_id(1)  # position in [0, T)
    k_offsets = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    src_ptr = C_ptr + b * (T + P) * K + d * K + k_offsets
    dst_ptr = out_ptr + b * T * K + d * K + k_offsets
    vals = tl.load(src_ptr, mask=mask_k, other=0.0)
    tl.store(dst_ptr, vals, mask=mask_k)


@triton.jit
def _split_hidden_kernel(
    C_ptr,          # *fp32, [B, T+P, K]
    out_ptr,        # *fp32, [B, P, K]
    B: tl.constexpr,
    T: tl.constexpr,
    P: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)  # batch
    d = tl.program_id(1)  # position in [0, P)
    k_offsets = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Source row in C is at index T + d
    src_ptr = C_ptr + b * (T + P) * K + (T + d) * K + k_offsets
    dst_ptr = out_ptr + b * P * K + d * K + k_offsets
    vals = tl.load(src_ptr, mask=mask_k, other=0.0)
    tl.store(dst_ptr, vals, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the given run() function.
        - Concatenation is done in Triton.
        - Linear projection (matmul) is done in Triton.
        - Splitting is done in Triton.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "hidden_states and encoder_hidden_states must be 3D"
        assert process_weight.dim() == 2 and process_weight.shape[0] == process_weight.shape[1], "process_weight must be square [K, K]"
        B, P, K = hidden_states.shape
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == K
        T = encoder_hidden_states.shape[1]

        device = hidden_states.device
        dtype = torch.float32  # keep fp32 for correctness

        # Ensure inputs are contiguous and fp32
        encoder = encoder_hidden_states.to(dtype).contiguous()
        hidden = hidden_states.to(dtype).contiguous()
        W = process_weight.to(dtype).contiguous()  # [K, K]

        # 1) Concatenate in Triton: Acat [B, T+P, K]
        total = T + P
        Acat = torch.empty((B, total, K), device=device, dtype=dtype)

        # Tile across K; choose 128 for good throughput
        BLOCK_K = 128
        grid_concat = (B, total, triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder, hidden, Acat,
            B, T, P, K,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM in Triton: Acat [M, K] x W [K, K] -> C_flat [M, K]
        M = B * total
        C_flat = torch.empty((M, K), device=device, dtype=dtype)

        # Tile sizes; start with 64x64x64
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_gemm](
            Acat.view(M, K), W, C_flat,
            B, T, P, K,
            M=M,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Reshape C_flat to [B, T+P, K]
        C = C_flat.view(B, total, K)

        # 4) Split in Triton to produce outputs
        processed_encoder = torch.empty((B, T, K), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=dtype)

        # Choose BLOCK_K for splitting kernels
        BLOCK_K_split = 128
        grid_e = (B, T, triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        grid_i = (B, P, triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_i](
            C, processed_hidden,
            B, T, P, K,
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
