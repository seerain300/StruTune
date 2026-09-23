import torch
import triton
import triton.language as tl


@triton.jit
def _concat_kernel(
    encoder_ptr,  # *ptr to [B, T, K]
    hidden_ptr,   # *ptr to [B, P, K]
    out_ptr,      # *ptr to [B, T+P, K]
    B, T, P, K,
    stride_e_b, stride_e_t, stride_e_k,
    stride_h_b, stride_h_p, stride_h_k,
    stride_o_b, stride_o_l, stride_o_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T+P, cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Decide source: encoder if l < T, else hidden[b, l-T, :]
    is_encoder = l < T

    # Compute source offsets
    # For encoder: e_offsets = b*stride_e_b + l*stride_e_t + k_offsets*stride_e_k
    # For hidden: h_offsets = b*stride_h_b + (l - T)*stride_h_p + k_offsets*stride_h_k
    e_offsets = b * stride_e_b + l * stride_e_t + k_offsets * stride_e_k
    h_offsets = b * stride_h_b + (l - T) * stride_h_p + k_offsets * stride_h_k

    # Destination offsets
    out_offsets = b * stride_o_b + l * stride_o_l + k_offsets * stride_o_k

    # Load based on condition; only one branch is active per program
    if is_encoder:
        vals = tl.load(encoder_ptr + e_offsets, mask=mask_k, other=0.0)
    else:
        vals = tl.load(hidden_ptr + h_offsets, mask=mask_k, other=0.0)

    # Store
    tl.store(out_ptr + out_offsets, vals, mask=mask_k)


@triton.jit
def _matmul_kernel(
    A_ptr,  # *ptr to [M, K], M = B*(T+P)
    W_ptr,  # *ptr to [K, K]
    C_ptr,  # *ptr to [M, K]
    B, T, P, K,  # shapes (for M_total)
    stride_A_m, stride_A_k,
    stride_W_k, stride_W_n,
    stride_C_m, stride_C_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(M, BLOCK_M), cdiv(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    M_total = B * (T + P)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M_total
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_k
        A_mask = mask_m[:, None] & mask_k[None, :]
        A = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # W tile: [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_offsets[:, None] * stride_W_k + n_offsets[None, :] * stride_W_n
        W_mask = mask_k[:, None] & mask_n[None, :]
        W = tl.load(W_ptrs, mask=W_mask, other=0.0)

        acc += tl.dot(A, W)

    # Store C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_k
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr,  # *ptr to [B, T+P, K]
    out_ptr,  # *ptr to [B, T, K]
    B, T, K,
    stride_C_b, stride_C_l, stride_C_k,
    stride_O_b, stride_O_l, stride_O_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T, cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    src_offsets = b * stride_C_b + l * stride_C_l + k_offsets * stride_C_k
    dst_offsets = b * stride_O_b + l * stride_O_l + k_offsets * stride_O_k

    vals = tl.load(C_ptr + src_offsets, mask=mask_k, other=0.0)
    tl.store(out_ptr + dst_offsets, vals, mask=mask_k)


@triton.jit
def _split_hidden_kernel(
    C_ptr,  # *ptr to [B, T+P, K]
    out_ptr,  # *ptr to [B, P, K]
    B, T, P, K,
    stride_C_b, stride_C_l, stride_C_k,
    stride_O_b, stride_O_l, stride_O_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, P, cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    p = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    src_offsets = b * stride_C_b + (T + p) * stride_C_l + k_offsets * stride_C_k
    dst_offsets = b * stride_O_b + p * stride_O_l + k_offsets * stride_O_k

    vals = tl.load(C_ptr + src_offsets, mask=mask_k, other=0.0)
    tl.store(out_ptr + dst_offsets, vals, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate along sequence dim using Triton kernel.
        - Apply GEMM using Triton matmul kernel.
        - Split outputs using Triton kernels.
        """
        device = hidden_states.device
        B = hidden_states.shape[0]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        T = encoder_hidden_states.shape[1]

        # Ensure float32 and contiguity
        hidden = hidden_states.contiguous().to(torch.float32)      # [B, P, K]
        encoder = encoder_hidden_states.contiguous().to(torch.float32)  # [B, T, K]
        W = process_weight.contiguous().to(torch.float32)          # [K, K]

        # 1) Concatenate via Triton: Acat [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=device, dtype=torch.float32)

        BLOCK_K = 64
        grid_concat = (B, L, triton.cdiv(K, BLOCK_K))
        _concat_kernel[grid_concat](
            encoder, hidden, Acat,
            B, T, P, K,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat [B*(T+P), K] @ W [K, K] -> C_flat [B*(T+P), K]
        M = B * L
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K_GEMM = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_gemm](
            Acat, W, C_flat,
            B, T, P, K,
            Acat.stride(0), Acat.stride(2),  # A strides: m=0, k=2
            W.stride(0), W.stride(1),        # W strides: k=0, n=1
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2
        )

        # 3) Reshape and split: processed [B, T+P, K] -> encoder and hidden
        processed = C_flat.view(B, L, K)

        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        BLOCK_K_split = 64
        grid_e = (B, T, triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_e](
            processed, processed_encoder,
            B, T, K,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        grid_h = (B, P, triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_h](
            processed, processed_hidden,
            B, T, P, K,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
