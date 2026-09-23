import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    enc_ptr,       # *fp32, [B, T, K]
    hid_ptr,       # *fp32, [B, P, K]
    out_ptr,       # *fp32, [B, L, K] where L = T + P
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(L, BLOCK_L), cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    l_block = tl.program_id(1)
    k_block = tl.program_id(2)

    L = T + P

    l = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_l = l < L
    mask_k = k < K

    # Select source tensor based on l < T
    is_from_enc = l < T
    enc_offsets = b * (T * K) + l * K + k
    hid_offsets = b * (P * K) + (l - T) * K + k

    # Build masks for loads
    enc_mask = (mask_l & is_from_enc)[:, None] & mask_k[None, :]
    hid_mask = (mask_l & (~is_from_enc))[:, None] & mask_k[None, :]

    # Load from enc where applicable and from hid where applicable
    enc_vals = tl.load(enc_ptr + enc_offsets, mask=enc_mask, other=0.0)
    hid_vals = tl.load(hid_ptr + hid_offsets, mask=hid_mask, other=0.0)
    vals = tl.where(is_from_enc[:, None], enc_vals, hid_vals)

    # Store into Acat[b, l, :]
    out_offsets = b * (L * K) + l * K + k
    store_mask = mask_l[:, None] & mask_k[None, :]
    tl.store(out_ptr + out_offsets, vals, mask=store_mask)


@triton.jit
def _gemm_btl_kernel(
    A_ptr,         # *fp32, [M, K], M = B * (T+P)
    W_ptr,         # *fp32, [K, N], N = K (process_weight.T)
    C_ptr,         # *fp32, [B, L, N] but we will index via m = b*(L*N) + l*N + n
    B, M, N,       # runtime ints
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid over (batch, row-tiles, col-tiles)
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    # Row and column indices for this tile
    m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)           # [BLOCK_M]
    n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)           # [BLOCK_N]
    mask_m = m < M
    mask_n = n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k_start in range(0, N, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)                # [BLOCK_K]
        mask_k = k < N

        # Load A tile: A is [M, K], contiguous across k
        A_offsets = m[:, None] * N + k[None, :]            # [BLOCK_M, BLOCK_K]
        A_mask = mask_m[:, None] & mask_k[None, :]
        A_tile = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile: W is [K, N], contiguous across n
        W_offsets = k[:, None] * N + n[None, :]            # [BLOCK_K, BLOCK_N]
        W_mask = mask_k[:, None] & mask_n[None, :]
        W_tile = tl.load(W_ptr + W_offsets, mask=W_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(A_tile, W_tile)                      # [BLOCK_M, BLOCK_N]

    # Store results to C[b, m, n]
    # We store using flat indexing: m = b*(L*N) + l*N + n, where L = M / B (if B=1) or arbitrary. Here, B is not used in addressing directly.
    # However, M = B*(L), and C is [B, L, N]. We pass C_ptr as flat, and here b is the first grid dimension. To keep kernel simple, we assume external host sets C_ptr appropriately.
    # Since Triton kernel doesn't have B as an index, we rely on host to pass C_ptr as [B, L, N] and write using b = tl.program_id(0).
    # But Triton kernel doesn't have access to b here; so we pass C_ptr as [M, N] and construct offsets accordingly.
    # To be explicit: we compute C offsets using m and n. We assume C_ptr is [M, N], so offsets are m * N + n.
    C_offsets = m[:, None] * N + n[None, :]                # [BLOCK_M, BLOCK_N]
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + C_offsets, acc, mask=C_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr,          # *fp32, [M, N], M = B*(T+P), N = K
    out_ptr,        # *fp32, [B, T, N]
    B, M, N, T,     # runtime ints
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(T, BLOCK_M), cdiv(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)  # over T
    n_block = tl.program_id(2)

    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)        # rows in [0, T)
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)        # cols in [0, N)
    mask_m = m < T
    mask_n = n < N

    # Source offsets in C: row indices in [0, T)
    src_rows = m
    C_offsets = src_rows[:, None] * N + n[None, :]       # [BLOCK_M, BLOCK_N]
    mask = mask_m[:, None] & mask_n[None, :]

    vals = tl.load(C_ptr + C_offsets, mask=mask, other=0.0)

    # Destination offsets in out: [B, T, N]
    dst_base = b * (T * N)
    dst_offsets = dst_base + src_rows[:, None] * N + n[None, :]
    tl.store(out_ptr + dst_offsets, vals, mask=mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr,          # *fp32, [M, N], M = B*(T+P), N = K
    out_ptr,        # *fp32, [B, P, N]
    B, M, N, P,     # runtime ints
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(P, BLOCK_M), cdiv(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)  # over P (rows in [T, T+P))
    n_block = tl.program_id(2)

    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)        # rows in [T, T+P)
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)        # cols in [0, N)
    mask_m = m < P
    mask_n = n < N

    # Source offsets in C: row indices start at T
    src_rows = m + T
    C_offsets = src_rows[:, None] * N + n[None, :]       # [BLOCK_M, BLOCK_N]
    mask = mask_m[:, None] & mask_n[None, :]

    vals = tl.load(C_ptr + C_offsets, mask=mask, other=0.0)

    # Destination offsets in out: [B, P, N]
    dst_base = b * (P * N)
    dst_offsets = dst_base + m[:, None] * N + n[None, :]
    tl.store(out_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure float32 and contiguous
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, K)
        assert process_weight.shape == (K, K)
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32
        assert hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous() and process_weight.is_contiguous()

        device = hidden_states.device

        # 1) Concatenate along sequence dimension in Triton: [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=device, dtype=torch.float32)

        BLOCK_L = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B=B, T=T, P=P, K=K,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat [B, L, K] -> flat A [M, K], W [K, K] -> C [M, K], then we'll reshape/view back
        #   Here, we run GEMM on Acat directly via a kernel that treats Acat as [M, K] where M = B * L.
        M = B * L
        W = process_weight.t()  # [K, K]
        # Allocate flat C
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        # BLOCK sizes for GEMM
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_gemm), triton.cdiv(K, BLOCK_N_gemm))
        _gemm_btl_kernel[grid_gemm](
            Acat, W, C_flat,
            B=B, M=M, N=K,
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=2
        )

        # 3) Split into [B, T, K] and [B, P, K] using Triton
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        BLOCK_M_split = 64
        BLOCK_N_split = 64

        # For encoder slice: rows 0..T-1
        grid_e = (B, triton.cdiv(T, BLOCK_M_split), triton.cdiv(K, BLOCK_N_split))
        _split_encoder_kernel[grid_e](
            C_flat, processed_encoder,
            B=B, M=M, N=K, T=T,
            BLOCK_M=BLOCK_M_split, BLOCK_N=BLOCK_N_split,
            num_warps=4, num_stages=2
        )

        # For hidden slice: rows T..T+P-1
        grid_h = (B, triton.cdiv(P, BLOCK_M_split), triton.cdiv(K, BLOCK_N_split))
        _split_hidden_kernel[grid_h](
            C_flat, processed_hidden,
            B=B, M=M, N=K, P=P,
            BLOCK_M=BLOCK_M_split, BLOCK_N=BLOCK_N_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
