import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    encoder_ptr, hidden_ptr, output_ptr,
    B, T, P, K,
    encoder_stride_b, encoder_stride_t, encoder_stride_k,
    hidden_stride_b, hidden_stride_p, hidden_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, L, cdiv(K, BLOCK_K)), where L = T + P
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_block = tl.program_id(2)

    # K indices for this tile
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Determine source: first T rows from encoder, remaining from hidden
    is_encoder = l < T

    # Compute base pointers
    if is_encoder:
        src = encoder_ptr + b * encoder_stride_b + l * encoder_stride_t + k_offsets * encoder_stride_k
        # Store to output at index l
        dst = output_ptr + b * out_stride_b + l * out_stride_l + k_offsets * out_stride_k
        tl.store(dst, tl.load(src, mask=k_mask), mask=k_mask)
    else:
        src = hidden_ptr + b * hidden_stride_b + (l - T) * hidden_stride_p + k_offsets * hidden_stride_k
        dst = output_ptr + b * out_stride_b + l * out_stride_l + k_offsets * out_stride_k
        tl.store(dst, tl.load(src, mask=k_mask), mask=k_mask)


@triton.jit
def _split_kernel(
    input_ptr, output_ptr,
    B, T, S, K,  # S is either T or P depending on which stream we split
    input_stride_b, input_stride_s, input_stride_k,
    output_stride_b, output_stride_s, output_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, S, cdiv(K, BLOCK_K))
    b = tl.program_id(0)
    s = tl.program_id(1)
    k_block = tl.program_id(2)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    src = input_ptr + b * input_stride_b + s * input_stride_s + k_offsets * input_stride_k
    dst = output_ptr + b * output_stride_b + s * output_stride_s + k_offsets * output_stride_k

    tl.store(dst, tl.load(src, mask=k_mask), mask=k_mask)


@triton.jit
def _matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B, M, K, N,
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A_tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # W_tile: [BLOCK_K, BLOCK_N], note W is [K, N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, w)

    # Store results to C
    c_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate hidden streams via Triton.
        - Apply linear projection via Triton GEMM.
        - Split back via Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L = T + P

        # 1) Concatenate encoder and hidden along sequence dimension using Triton
        # encoder_hidden_states: [B, T, K]
        # hidden_states: [B, P, K]
        # output: Acat [B, L, K]
        Acat = torch.empty((B, L, K), device=hidden_states.device, dtype=torch.float32)

        BLOCK_K = 64
        grid_concat = (B, L, triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection via Triton GEMM: Acat [B*L, K] @ W [K, K]
        # Treat Acat as [M, K], W as [K, N], here N == K.
        M = B * L
        # Ensure process_weight is [K, K] as expected
        W = process_weight  # [K, K]
        C_flat = torch.empty((M, K), device=hidden_states.device, dtype=torch.float32)

        # 3D grid: (B, cdiv(M, BLOCK_M), cdiv(K, BLOCK_N))
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K_reduce = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_gemm](
            Acat, W, C_flat,
            B, M, K, K,
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            W.stride(0), W.stride(1),  # W is [K, K]
            C_flat.stride(0), C_flat.stride(1), C_flat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K_reduce,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape C_flat to [B, L, K]
        C = C_flat.view(B, L, K)

        # 4) Split into encoder and hidden via Triton
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        BLOCK_K_split = 64
        grid_e = (B, T, triton.cdiv(K, BLOCK_K_split))
        _split_kernel[grid_e](
            C, processed_encoder,
            B, T, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2,
        )

        grid_h = (B, P, triton.cdiv(K, BLOCK_K_split))
        _split_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
