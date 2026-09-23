import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D], float32
    H_ptr,        # hidden_states: [B, I, D], float32
    C_ptr,        # concatenated output: [B, T+I, D], float32
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seqlen_stride, C_d_stride,
):
    # Grid: (B, T+I)
    pid_b = tl.program_id(0)
    pid_seq = tl.program_id(1)
    t_total = T + I
    stream = pid_seq // t_total
    pos = pid_seq % t_total
    if stream == 1:
        pos = pos - T

    src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride if stream == 0 else H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_seq * C_seqlen_stride

    # Copy contiguous vector of length D
    # E/H tensors are [*, *, D], D is the last dimension, so E_d_stride = H_d_stride = 1 for contiguous.
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)
        tl.store(dst_ptr + d * C_d_stride, val)


@triton.jit
def _batched_gemm_kernel(
    X_ptr,        # input X: [B, M_total, K] where M_total = T + I
    W_ptr,        # process_weight: [K, N] (note: using W in its native [K, N] layout)
    Y_ptr,        # output Y: [B, M_total, N]
    B, M_total, K, N,
    X_b_stride, X_m_stride, X_k_stride,
    W_k_stride, W_n_stride,
    Y_b_stride, Y_m_stride, Y_n_stride,
    BLOCK_M: tl.constexpr,  # tile along M (rows)
    BLOCK_N: tl.constexpr,  # tile along N (cols)
    BLOCK_K: tl.constexpr,  # tile along K (inner dim)
):
    # program ids
    pid_b = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # tiles along M
    pid_n = tl.program_id(2)  # tiles along N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for boundaries
    m_mask = m_offsets < M_total
    n_mask = n_offsets < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load X tile: shape [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + pid_b * X_b_stride + m_offsets[:, None] * X_m_stride + k_offsets[None, :] * X_k_stride
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_k_stride + n_offsets[None, :] * W_n_stride
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

    # Store result Y tile
    y_ptrs = Y_ptr + pid_b * Y_b_stride + m_offsets[:, None] * Y_m_stride + n_offsets[None, :] * Y_n_stride
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates along sequence dimension in Triton.
        - Computes linear projection (matmul without bias) in Triton.
        - Splits outputs back into encoder and image streams.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure float32 for deterministic behavior and use contiguous tensors for Triton
        dtype = torch.float32
        device = hidden_states.device
        assert hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16), "Inputs must be float32/float16/bfloat16"
        assert encoder_hidden_states.device == device and process_weight.device == device, "All tensors must be on the same device"

        # Make contiguous and cast to float32 for compute (original example uses float32)
        E = encoder_hidden_states.contiguous().to(dtype)
        H = hidden_states.contiguous().to(dtype)
        W = process_weight.contiguous().to(dtype)

        B, T, D = E.shape
        Bi, I, Di = H.shape
        assert B == Bi, "Batch sizes must match"
        assert D == Di, "Hidden dimensions must match"
        assert W.shape[0] == D and W.shape[1] == D, "process_weight must be [hidden_dim, hidden_dim] == [D, D] in this setup"

        # 1) Concatenate in Triton: C [B, T+I, D]
        I_curr = I  # to avoid confusion with python keyword 'in'
        M_total = T + I_curr
        C = torch.empty((B, M_total, D), device=device, dtype=dtype)

        grid_concat = (B, M_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I_curr, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Compute Y = C @ W^T in Triton (note: W is [D, D]; we use W in [K, N] = [D, D])
        # Output Y: [B, M_total, D]
        Y = torch.empty((B, M_total, D), device=device, dtype=dtype)

        # Launch Triton batched GEMM: grid over (B, tiles_M, tiles_N)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _batched_gemm_kernel[grid](
            C, W, Y,
            B, M_total, D, D,  # K=N=D
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),  # K stride, N stride for W
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split Y into two outputs
        processed_encoder = Y[:, :T, :]  # [B, T, D]
        processed_hidden = Y[:, T:, :]   # [B, I, D]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
