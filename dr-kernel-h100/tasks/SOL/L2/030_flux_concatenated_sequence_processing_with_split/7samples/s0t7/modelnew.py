import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    out_ptr,         # *fp32, output [B, M, K], M = T + I
    x1_ptr,          # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,          # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr, # batch size
    T: tl.constexpr, # text_seq_len
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden_dim
):
    # Grid: (B, tiles over M = T+I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    BLOCK_M = 128
    BLOCK_K = 64

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # total sequence positions
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden_dim indices

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # For each m in the tile, determine source tensor: encoder (x1) if m < T, else image (x2)
    is_encoder = m_offsets < T

    # Base pointers for each tensor (contiguous layout assumed: [B, dim0, K])
    # x1 strides: s0 = T*K, s1 = K, s2 = 1
    # x2 strides: s0 = I*K, s1 = K, s2 = 1
    # out strides: s0 = M*K, s1 = K, s2 = 1
    # Note: Triton pointers are byte-addressed; element-wise addressing uses strides in elements.

    # We will write out[b, m, k] using strides. Since tensors are contiguous in the example,
    # we can directly use k_offsets as column stride.

    # Loop over K tile
    for k_idx in range(0, BLOCK_K):
        k = k_offsets[k_idx]
        if not mask_k[k_idx]:
            continue
        # Prepare row indices
        row_encoder = tl.where(is_encoder, m_offsets, 0)  # mask will avoid out-of-range reads
        row_image = m_offsets - T

        # Load from x1 for encoder rows, else from x2
        # Masked loads: if not mask_m, we skip
        # Create per-row pointers (vector of length BLOCK_M) for each tensor
        x1_row_ptrs = x1_ptr + b * (T * K) + row_encoder * K + k
        x2_row_ptrs = x2_ptr + b * (I * K) + (row_image) * K + k

        # Build masks for each row
        mask_m_encoder = (mask_m) & (is_encoder)
        mask_m_image = (mask_m) & (~is_encoder)

        # Load values
        v_encoder = tl.load(x1_row_ptrs, mask=mask_m_encoder, other=0.0)
        v_image = tl.load(x2_row_ptrs, mask=mask_m_image, other=0.0)

        # Combine: where is_encoder, v_encoder else v_image
        v = tl.where(is_encoder, v_encoder, v_image)

        # Store to out[b, m, k]
        out_ptrs = out_ptr + b * ((T + I) * K) + m_offsets * K + k
        tl.store(out_ptrs, v, mask=mask_m)


@triton.jit
def batched_matmul_kernel(
    out_ptr, A_ptr, W_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + b * (M * K) + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = W_ptr + offs_k[:, None] * N + offs_n[None, :]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, w)

    out_ptrs = out_ptr + b * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim into A [B, T+I, K].
        2) Compute C = A @ process_weight.T via Triton matmul: C [B, T+I, K].
        3) Return split streams: (C[:, :T, :], C[:, T:, :])
        """
        # Ensure contiguity and dtype (compute in fp32 for robustness)
        x1 = encoder_hidden_states.contiguous()
        x2 = hidden_states.contiguous()
        W = process_weight.contiguous()

        B = x1.shape[0]
        T = x1.shape[1]
        I = x2.shape[1]
        K = x1.shape[2]
        assert x2.shape[2] == K, "hidden_dim must match between encoder_hidden_states and hidden_states"
        assert W.shape[0] == K and W.shape[1] == K, "process_weight must be [hidden_dim, hidden_dim]"

        device = x1.device

        # 1) Concatenate sequences into A [B, M, K] using Triton
        M = T + I
        A = torch.empty((B, M, K), device=device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_kernel[grid_concat](
            A, x1, x2, B, T, I, K,
            num_warps=4, num_stages=2
        )

        # 2) Batched GEMM: C = A @ W (W is [K, K], output [B, M, K])
        C = torch.empty((B, M, K), device=device, dtype=torch.float32)

        BLOCK_M_G = 128
        BLOCK_N_G = 128
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W, B, M, K, K,
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2
        )

        # 3) Split back into separate streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtype if needed (inputs are typically float32)
        if processed_encoder.dtype != x1.dtype:
            processed_encoder = processed_encoder.to(x1.dtype)
        if processed_hidden.dtype != x2.dtype:
            processed_hidden = processed_hidden.to(x2.dtype)

        return processed_encoder, processed_hidden