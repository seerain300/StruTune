import torch
import triton
import triton.language as tl


@triton.jit
def _concat_kernel(
    encoder_ptr,        # *fp32, [B, T, K]
    hidden_ptr,         # *fp32, [B, P, K]
    out_ptr,            # *fp32, [B, T+P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, cdiv(T+P, BLOCK_L), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    tile_l = tl.program_id(1)
    tile_k = tl.program_id(2)

    l = tile_l * BLOCK_L + tl.arange(0, BLOCK_L)  # concatenated sequence positions
    k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)  # feature dimension
    mask_l = l < (T + P)
    mask_k = k < K

    # Determine source for each l: if l < T -> encoder, else hidden
    src = tl.where(l < T, 0, 1)  # 0 for encoder, 1 for hidden

    # Compute pointers for encoder and hidden rows
    # encoder: [B, T, K]
    enc_row_ptr = encoder_ptr + b * T * K + l * K  # + l*stride(1)
    # hidden: [B, P, K]
    hid_row = l - T
    hid_row_mask = (l >= T) & (hid_row < P)
    hid_row_ptr = hidden_ptr + b * P * K + hid_row * K

    # Build mask for loads: only valid l and k
    enc_mask = mask_l & (l < T)
    hid_mask = mask_l & (~enc_mask) & mask_k

    # Load encoder rows (if l < T) and hidden rows (if l >= T)
    # We'll use tl.load with masks; for invalid, load zeros.
    enc_vals = tl.load(enc_row_ptr[:, None] + k[None, :], mask=enc_mask[:, None] & mask_k[None, :], other=0.0)
    hid_vals = tl.load(hid_row_ptr[:, None] + k[None, :], mask=hid_mask[:, None], other=0.0)
    # Combine: for l >= T, use hid_vals; otherwise use enc_vals
    vals = tl.where(l < T, enc_vals, hid_vals)

    # Store to out: [B, T+P, K]
    out_row_ptr = out_ptr + b * (T + P) * K + l * K
    store_mask = mask_l[:, None] & mask_k[None, :]
    tl.store(out_row_ptr[:, None] + k[None, :], vals, mask=store_mask)


@triton.jit
def _matmul_rows_kernel(
    A_ptr,       # *fp32, [M, K] where M = B*(T+P)
    W_ptr,       # *fp32, [K, K]
    C_ptr,       # *fp32, [M, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    M: tl.constexpr,  # M = B*(T+P)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # typically equals K or a tile of K
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, cdiv(M, BLOCK_M), cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices in [0, M)
    n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)  # column indices in [0, K)

    mask_m = m < M
    mask_n = n < K

    # Compute corresponding batch and sequence indices for A rows
    total = T + P
    b_idx = m // total
    r = m % total  # position within concatenated sequence

    # A is [M, K], row index is m
    A_row_ptr = A_ptr + m * K  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_row_ptr[:, None] + k[None, :]
        A_vals = tl.load(A_tile_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N], W is [K, K]
        W_tile_ptr = W_ptr + k[:, None] * K + n[None, :]
        W_vals = tl.load(W_tile_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(A_vals, W_vals)

    # Store C tile: C is [M, K], row index m, col n
    C_tile_ptr = C_ptr + m[:, None] * K + n[None, :]
    tl.store(C_tile_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _split_encoder_kernel(
    C_ptr,              # *fp32, [M, K] where M = B*(T+P)
    out_ptr,            # *fp32, [B, T, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, cdiv(T, BLOCK_T), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    tile_t = tl.program_id(1)
    tile_k = tl.program_id(2)

    t = tile_t * BLOCK_T + tl.arange(0, BLOCK_T)
    k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_t = t < T
    mask_k = k < K

    # Row indices in C: rows 0..T-1
    m = b * (T + P) + t

    C_row_ptr = C_ptr + m[:, None] * K + k[None, :]
    C_vals = tl.load(C_row_ptr, mask=mask_t[:, None] & mask_k[None, :], other=0.0)

    # Store to output [B, T, K]
    out_row_ptr = out_ptr + b * T * K + t[:, None] * K + k[None, :]
    tl.store(out_row_ptr, C_vals, mask=mask_t[:, None] & mask_k[None, :])


@triton.jit
def _split_hidden_kernel(
    C_ptr,              # *fp32, [M, K] where M = B*(T+P)
    out_ptr,            # *fp32, [B, P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    M: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, cdiv(P, BLOCK_P), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    tile_p = tl.program_id(1)
    tile_k = tl.program_id(2)

    p = tile_p * BLOCK_P + tl.arange(0, BLOCK_P)
    k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_p = p < P
    mask_k = k < K

    # Row indices in C: rows T..T+P-1
    m = b * (T + P) + T + p

    C_row_ptr = C_ptr + m[:, None] * K + k[None, :]
    C_vals = tl.load(C_row_ptr, mask=mask_p[:, None] & mask_k[None, :], other=0.0)

    # Store to output [B, P, K]
    out_row_ptr = out_ptr + b * P * K + p[:, None] * K + k[None, :]
    tl.store(out_row_ptr, C_vals, mask=mask_p[:, None] & mask_k[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate along sequence using _concat_kernel
        - Linear projection using _matmul_rows_kernel
        - Split into encoder and hidden streams using _split_encoder_kernel and _split_hidden_kernel
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B, T, K = encoder_hidden_states.shape
        P = hidden_states.shape[1]
        # Ensure contiguous tensors
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        W = process_weight.contiguous()  # [K, K]

        # 1) Concatenate encoder and hidden along sequence using Triton
        Acat = torch.empty((B, T + P, K), device=hidden.device, dtype=torch.float32)
        BLOCK_L = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(T + P, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concat_kernel[grid_concat](
            encoder, hidden, Acat,
            B, T, P, K,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat [B*(T+P), K] @ W.T [K, K] -> C_flat [B*(T+P), K]
        M = B * (T + P)
        C = torch.empty((M, K), device=hidden.device, dtype=torch.float32)

        # Choose block sizes; K is often power of two, but we use masks for generality
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_rows_kernel[grid_gemm](
            Acat.view(M, K), W, C,
            B, T, P, K,
            M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split C into processed_encoder and processed_hidden using Triton
        processed_encoder = torch.empty((B, T, K), device=hidden.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden.device, dtype=torch.float32)

        BLOCK_T = 64
        BLOCK_Ks = 64
        grid_e = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_Ks))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, P, K, M,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_Ks,
            num_warps=4, num_stages=2
        )

        BLOCK_P = 64
        grid_h = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(K, BLOCK_Ks))
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K, M,
            BLOCK_P=BLOCK_P, BLOCK_K=BLOCK_Ks,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
