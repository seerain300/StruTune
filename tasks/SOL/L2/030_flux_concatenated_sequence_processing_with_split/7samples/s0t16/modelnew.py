import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    OUT_ptr,          # *fp32, output A: [B, M, K], M = T + I
    X1_ptr,           # *fp32, encoder_hidden_states: [B, T, K]
    X2_ptr,           # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr,  # batch size
    T: tl.constexpr,  # text_seq_len
    I: tl.constexpr,  # img_seq_len
    K: tl.constexpr,  # hidden_dim
    OUT_s0, OUT_s1, OUT_s2,
    X1_s0, X1_s1, X1_s2,
    X2_s0, X2_s1, X2_s2,
    BLOCK_M: tl.constexpr,  # tile along M (sequence)
    BLOCK_K: tl.constexpr,  # tile along K (hidden)
):
    # Grid: (B, tiles over M = T+I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # indices in concatenated sequence
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden_dim indices

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine if this m corresponds to encoder (x1) or image (x2)
    is_encoder = m_offsets < T

    # Initialize accumulators for each k element in the tile
    # We will store a vector over k_offsets for each m in the block.
    # For masked m, store zeros.
    for m_idx in range(0, BLOCK_M):
        m = m_offsets[m_idx]
        # If m is out of bounds, continue
        if not mask_m[m_idx]:
            continue
        # Compute the row base pointers
        # OUT row base for this (b, m)
        out_row_ptr = OUT_ptr + b * OUT_s0 + m * OUT_s1

        # We need to load from either X1 or X2 depending on whether m < T.
        # X1: [b, m, k]
        # X2: [b, m - T, k]
        # Compute addresses for each k in the tile
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k_idx in range(0, BLOCK_K):
            k = k_offsets[k_idx]
            if not mask_k[k_idx]:
                continue
            if is_encoder[m_idx]:
                x1_addr = X1_ptr + b * X1_s0 + m * X1_s1 + k * X1_s2
                val = tl.load(x1_addr, mask=True, other=0.0)
            else:
                x2_m = m - T
                x2_addr = X2_ptr + b * X2_s0 + x2_m * X2_s1 + k * X2_s2
                val = tl.load(x2_addr, mask=True, other=0.0)
            acc[k_idx] = val

        # Store acc into OUT row at positions k_offsets
        out_addr = out_row_ptr + k_offsets * OUT_s2
        tl.store(out_addr, acc, mask=mask_k)


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    A_s0, A_s1, A_s2,       # strides for A: [B, M, K]
    W_s0, W_s1,             # strides for W: [K, K]
    C_s0, C_s1, C_s2,       # strides for C: [B, M, K]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3D launch grid: (batch, tiles along M, tiles along N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A / output rows
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # output cols (K)

    mask_m = m_offsets < M
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets_k < K

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets_k[None, :] * A_s2
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets_k[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: acc += a @ w  -> (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) => (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, w)

    # Store results
    c_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension into A [B, M, K].
        2) Compute C = A @ process_weight via Triton batched matmul.
        3) Split C into processed_encoder [B, T, K] and processed_hidden [B, I, K].
        """
        # Extract shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        # We will do compute in float32 in kernels. Cast inputs to float32 for kernels.
        X1 = encoder_hidden_states.contiguous()
        X2 = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Allocate output A (concatenated) in float32
        M = T + I
        A = torch.empty((B, M, K), dtype=torch.float32, device=X1.device)

        # Launch concat kernel
        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A, X1, X2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            X1.stride(0), X1.stride(1), X1.stride(2),
            X2.stride(0), X2.stride(1), X2.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Allocate C for output of GEMM
        C = torch.empty((B, M, K), dtype=torch.float32, device=X1.device)

        # Cast weight to fp32 for kernel (it's already fp32 in most setups; ensure contiguous)
        W_fp32 = W.to(torch.float32).contiguous()

        # Launch batched matmul kernel
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W_fp32,
            B, M, K,
            A.stride(0), A.stride(1), A.stride(2),
            W_fp32.stride(0), W_fp32.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2
        )

        # Split outputs back into encoder and image streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast outputs back to original dtype if needed
        # The original run returns tensors with the same dtype as inputs (float32 in provided evals)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden