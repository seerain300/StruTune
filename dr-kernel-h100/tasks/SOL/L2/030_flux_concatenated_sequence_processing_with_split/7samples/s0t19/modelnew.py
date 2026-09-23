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
    BLOCK_M: tl.constexpr,  # tile along sequence (M)
    BLOCK_K: tl.constexpr,  # tile along hidden dim (K)
):
    # Grid: (B, tiles over M = T+I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence positions in concatenated
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden_dim indices

    # Masks for bounds
    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine if m belongs to encoder (first T) or image (next I)
    is_encoder = m_offsets < T

    # Compute output addresses for each m
    out_ptrs = OUT_ptr + b * OUT_s0 + m_offsets[:, None] * OUT_s1 + k_offsets[None, :] * OUT_s2

    # For each m (segment), we need to load from X1 or X2
    # Loop over BLOCK_M in the m-dimension for this tile
    for mm in range(0, BLOCK_M):
        m_idx = m_offsets[mm]
        # If m_idx is out of bounds, skip
        if not mask_m[mm]:
            continue

        # Compute input addresses for X1 and X2
        x1_ptr = X1_ptr + b * X1_s0 + m_idx * X1_s1 + k_offsets[None, :] * X1_s2
        x2_ptr = X2_ptr + b * X2_s0 + (m_idx - T) * X2_s1 + k_offsets[None, :] * X2_s2  # (m_idx - T) maps image positions

        # Choose source: encoder if is_encoder[mm], else image
        src = tl.where(is_encoder[mm], x1_ptr, x2_ptr)

        # Load a vector along K (masked)
        vals = tl.load(src, mask=mask_k, other=0.0)

        # Store to output [b, m_idx, k_offsets]
        out_vec_ptr = OUT_ptr + b * OUT_s0 + m_idx * OUT_s1 + k_offsets[None, :] * OUT_s2
        tl.store(out_vec_ptr, vals, mask=mask_k)


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

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A (sequence positions)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns (hidden dim)

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A_tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W_tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store results to C
    c_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim (dim=1)
        2) Apply linear projection: processed = A @ process_weight.T
        3) Split back into processed_encoder and processed_hidden
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # 1) Concatenate along sequence dimension using Triton
        A = torch.empty((B, M, K), dtype=torch.float32, device=hidden_states.device)

        # Ensure inputs are contiguous and in float32 for Triton kernels
        x1 = encoder_hidden_states.contiguous().to(torch.float32)
        x2 = hidden_states.contiguous().to(torch.float32)

        # Kernel launch config for concat
        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B=B, T=T, I=I, K=K,
            OUT_s0=A.stride(0), OUT_s1=A.stride(1), OUT_s2=A.stride(2),
            X1_s0=x1.stride(0), X1_s1=x1.stride(1), X1_s2=x1.stride(2),
            X2_s0=x2.stride(0), X2_s1=x2.stride(1), X2_s2=x2.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # 2) Batched GEMM: C = A @ W, where W is [K, K]
        W = process_weight.contiguous().to(torch.float32)  # process_weight is [K, K] as in the original code
        C = torch.empty((B, M, K), dtype=torch.float32, device=hidden_states.device)

        # Kernel launch config for matmul
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B=B, M=M, K=K,
            A_s0=A.stride(0), A_s1=A.stride(1), A_s2=A.stride(2),
            W_s0=W.stride(0), W_s1=W.stride(1),
            C_s0=C.stride(0), C_s1=C.stride(1), C_s2=C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast outputs back to the original input dtype (match reference behavior)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden