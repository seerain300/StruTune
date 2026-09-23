import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    C_ptr,        # concatenated output: [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_s_stride, C_d_stride,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, T+I)
    pid_b = tl.program_id(0)
    pid_pos = tl.program_id(1)
    t_total = T + I

    # Determine source stream: 0 => encoder, 1 => hidden
    stream = pid_pos // t_total
    pos = pid_pos % t_total
    if stream == 1:
        pos = pos - T  # map position within hidden sequence

    # Compute base pointers
    if stream == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_s_stride

    # Copy a vector of length D
    for d in range(0, BLOCK_D):
        d_val = d
        mask = d_val < D
        val = tl.load(src_ptr + d_val * E_d_stride, mask=mask, other=0.0)
        tl.store(dst_ptr + d_val * C_d_stride, val, mask=mask)


@triton.jit
def _batched_matmul_kernel(
    X_ptr,  # [B, M, K] input matrix (concatenated [B, T+I, D])
    W_ptr,  # [K, N] weight matrix (process_weight [D, D], transposed [D, D] already)
    Y_ptr,  # [B, M, N] output
    B, M, N, K,
    X_b_stride, X_m_stride, X_k_stride,
    W_k_stride, W_n_stride,
    Y_b_stride, Y_m_stride, Y_n_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    m_offsets = m0 + tl.arange(0, BLOCK_M)
    n_offsets = n0 + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + pid_b * X_b_stride + m_offsets[:, None] * X_m_stride + k_offsets[None, :] * X_k_stride
        x_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_k_stride + n_offsets[None, :] * W_n_stride
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x, w)

    # Store Y tile
    y_ptrs = Y_ptr + pid_b * Y_b_stride + m_offsets[:, None] * Y_m_stride + n_offsets[None, :] * Y_n_stride
    y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run() function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Computes processed = concatenated @ process_weight.T using a Triton batched matmul kernel.
        - Splits processed into encoder and hidden streams (torch slicing).
        Returns (processed_encoder, processed_hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 for consistency."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, D), "encoder_hidden_states shape must be [batch, text_seq_len, hidden_dim]"
        assert hidden_states.shape == (B, I, D), "hidden_states shape must be [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (D, D), "process_weight must be [hidden_dim, hidden_dim]"

        # Ensure contiguous
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()  # [D, D], we will pass as [K, N] where K=N=D

        # 1) Concatenate in Triton: [B, T+I, D]
        total = T + I
        C = torch.empty((B, total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_D=64, num_warps=4, num_stages=2,
        )

        # 2) Compute processed = C @ W.T using Triton batched matmul
        # We need X [B, M, K] where M = total = T+I, K = D; W [K, N] where N = D.
        M = total
        K = D
        N = D
        Y = torch.empty((B, M, N), device=C.device, dtype=torch.float32)

        # Grid: (B, tiles over M, tiles over N)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _batched_matmul_kernel[grid_matmul](
            C, W, Y,
            B, M, N, K,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split result back into encoder and hidden streams using torch slicing (safe and fast)
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        return processed_encoder, processed_hidden