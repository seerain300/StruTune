import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    enc_ptr,     # *fp32, [B, T, K]
    hid_ptr,     # *fp32, [B, P, K]
    out_ptr,     # *fp32, [B, L, K], L = T + P
    B, T, P, K,  # int32
    BLOCK_L: tl.constexpr,  # tile along sequence (L = T+P)
    BLOCK_K: tl.constexpr,  # tile along feature dim
):
    # 3D grid: (batch, tiles along L, tiles along K)
    b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_k = tl.program_id(2)

    L = T + P
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)  # [BLOCK_L]
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]
    mask_l = l_offsets < L
    mask_k = k_offsets < K

    # Determine source: if l < T -> encoder, else -> hidden
    is_encoder = l_offsets < T
    # Compute element indices for enc and hid
    enc_idx = b * T * K + l_offsets * K + k_offsets[None, :]    # [BLOCK_L, BLOCK_K]
    hid_idx = b * P * K + (l_offsets - T) * K + k_offsets[None, :]  # [BLOCK_L, BLOCK_K]

    # Masks for loads
    enc_mask = mask_l[:, None] & mask_k[None, :]
    hid_mask = mask_l[:, None] & mask_k[None, :]
    # For elements where is_encoder is False, enc_mask should be False
    enc_mask = enc_mask & is_encoder[:, None]
    # For elements where is_encoder is True, hid_mask should be False
    hid_mask = hid_mask & (~is_encoder)[:, None]

    # Load values from appropriate source
    vals = tl.zeros((BLOCK_L, BLOCK_K), dtype=tl.float32)
    # Load from encoder where applicable
    vals += tl.load(enc_ptr + enc_idx, mask=enc_mask, other=0.0)
    # Load from hidden where applicable
    vals += tl.load(hid_ptr + hid_idx, mask=hid_mask, other=0.0)

    # Store to output Acat[b, l, k]
    out_idx = b * L * K + l_offsets * K + k_offsets[None, :]
    out_mask = mask_l[:, None] & mask_k[None, :]
    tl.store(out_ptr + out_idx, vals, mask=out_mask)


@triton.jit
def _gemm_kernel(
    A_ptr,   # *fp32, [M, K], M = B * (T+P)
    W_ptr,   # *fp32, [K, N], N = K (process_weight.T)
    C_ptr,   # *fp32, [M, N]
    B, M, K, N,  # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid over batch, rows tiles, cols tiles
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Each output tile corresponds to a specific batch
    # m = b * (T+P) + m_offsets if we had to decompose, but here A is flat [M, K]
    # We will compute for each m in m_offsets, across all n_offsets.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load A_tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * K + k_offsets[None, :]
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W_tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * N + n_offsets[None, :]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store to C[b, m, n]
    c_ptrs = C_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr,        # *fp32, [B, T+P, K]
    out_ptr,      # *fp32, [B, T, K]
    B, T, K,      # int32
    BLOCK_M: tl.constexpr,  # tile along rows (T)
    BLOCK_K: tl.constexpr,  # tile along features (K)
):
    # 3D grid over batch, rows tiles, cols tiles
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows in [0, T)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)   # features in [0, K)
    mask_m = m_offsets < T
    mask_k = k_offsets < K

    # Load from C at indices (b, m, k): C[b, m, k]
    c_ptrs = C_ptr + b * (T + P) * K + m_offsets[:, None] * K + k_offsets[None, :]
    c_mask = mask_m[:, None] & mask_k[None, :]
    vals = tl.load(c_ptrs, mask=c_mask, other=0.0)

    # Store to out at indices (b, m, k): out[b, m, k]
    out_ptrs = out_ptr + b * T * K + m_offsets[:, None] * K + k_offsets[None, :]
    tl.store(out_ptrs, vals, mask=c_mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr,        # *fp32, [B, T+P, K]
    out_ptr,      # *fp32, [B, P, K]
    B, T, P, K,   # int32
    BLOCK_M: tl.constexpr,  # tile along rows (P)
    BLOCK_K: tl.constexpr,  # tile along features (K)
):
    # 3D grid over batch, rows tiles, cols tiles
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows in [0, P)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)   # features in [0, K)
    mask_m = m_offsets < P
    mask_k = k_offsets < K

    # Load from C at indices (b, T+m, k): C[b, T+m, k]
    c_ptrs = C_ptr + b * (T + P) * K + (T + m_offsets)[:, None] * K + k_offsets[None, :]
    c_mask = mask_m[:, None] & mask_k[None, :]
    vals = tl.load(c_ptrs, mask=c_mask, other=0.0)

    # Store to out at indices (b, m, k): out[b, m, k]
    out_ptrs = out_ptr + b * P * K + m_offsets[:, None] * K + k_offsets[None, :]
    tl.store(out_ptrs, vals, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure dtype is fp32 (original code uses default fp32). If not, cast for safety.
        # We keep dtype consistent with original: if inputs are fp32, stay fp32. process_weight is fp32 typically.
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B, P, K = hidden_states.shape
        T = encoder_hidden_states.shape[1]
        Khs = encoder_hidden_states.shape[2]
        assert Khs == K, "hidden_dim mismatch between encoder and image inputs"
        W = process_weight  # [K, K], already [K, K]

        # 1) Concatenate in Triton: Acat [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=hidden_states.device, dtype=torch.float32)

        BLOCK_L = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM in Triton: Acat [B*L, K] @ W.T [K, K] -> C_flat [B*L, K]
        M = B * L
        N = K  # since W is [K, K]
        C_flat = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)

        # For simplicity, use BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _gemm_kernel[grid_gemm](
            Acat, W, C_flat,
            B, M, K, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Reshape C_flat to [B, L, K] and split in Triton
        C = C_flat.view(B, L, K)

        # Allocate outputs
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        # SPLIT: processed_encoder = C[:, :T, :]
        BLOCK_MT = 64
        BLOCK_KT = 64
        grid_e = (B, triton.cdiv(T, BLOCK_MT), triton.cdiv(K, BLOCK_KT))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            BLOCK_M=BLOCK_MT, BLOCK_K=BLOCK_KT,
            num_warps=4, num_stages=2
        )

        # SPLIT: processed_hidden = C[:, T:, :]
        BLOCK_MP = 64
        BLOCK_KP = 64
        grid_h = (B, triton.cdiv(P, BLOCK_MP), triton.cdiv(K, BLOCK_KP))
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K,
            BLOCK_M=BLOCK_MP, BLOCK_K=BLOCK_KP,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
