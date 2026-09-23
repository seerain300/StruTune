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
    C_b_stride, C_seqlen_stride, C_d_stride,
):
    # 3D grid: (batch, t_or_i, d_tile)
    pid_b = tl.program_id(0)
    pid_stream = tl.program_id(1)  # 0 => encoder, 1 => hidden
    pid_d = tl.program_id(2)

    t_total = T + I
    stream = pid_stream
    d_offsets = pid_d * D + tl.arange(0, D)
    # Compute row index in concatenated tensor
    row = stream * t_total
    # For stream 1 (hidden), we map to i in [0, I)
    # Here we pass stream directly as 0 or 1; row = 0..T-1 for encoder, T..T+I-1 for hidden
    # So when stream == 1, row += I
    row += I * (stream == 1)

    # Batch pointer offsets
    e_off = pid_b * E_b_stride
    c_off = pid_b * C_b_stride

    # Load and store vector of length D
    for d in range(0, D):
        e_ptr = E_ptr + e_off + row * E_t_stride + d * E_d_stride
        c_ptr = C_ptr + c_off + (row) * C_seqlen_stride + d * C_d_stride
        val = tl.load(e_ptr)
        tl.store(c_ptr, val)


@triton.jit
def _matmul_batched_kernel(
    X_ptr,        # input [B, M_total, D]
    W_ptr,        # weight [D, D]
    Y_ptr,        # output [B, M_total, D]
    B, M_total, D,
    X_b_stride, X_m_stride, X_d_stride,
    W_d0_stride, W_d1_stride,  # strides for W: [D, D]
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_M: tl.constexpr,     # tile size along M (rows of output per batch)
    BLOCK_N: tl.constexpr,     # tile size along N (output columns)
    BLOCK_K: tl.constexpr,     # tile size along K (input columns)
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in [0, M_total)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns in [0, D)
    offs_k = tl.arange(0, BLOCK_K)                    # reduction dim in [0, D)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, D, BLOCK_K):
        # Current K slice
        k_idx = k + offs_k  # [BLOCK_K]
        # Pointers for X: shape [B, M_total, D]
        x_ptrs = X_ptr + pid_b * X_b_stride + offs_m[:, None] * X_m_stride + k_idx[None, :] * X_d_stride
        # Pointers for W: shape [D, D]
        w_ptrs = W_ptr + k_idx[:, None] * W_d0_stride + offs_n[None, :] * W_d1_stride

        # Masks for boundary
        x_mask = (offs_m[:, None] < M_total) & (k_idx[None, :] < D)
        w_mask = (k_idx[:, None] < D) & (offs_n[None, :] < D)

        # Load tiles
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)     # [BLOCK_M, BLOCK_K]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)     # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x, w)

    # Write back results
    y_ptrs = Y_ptr + pid_b * Y_b_stride + offs_m[:, None] * Y_m_stride + offs_n[None, :] * Y_d_stride
    y_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < D)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _split_copy_kernel(
    Y_ptr,            # input processed concatenated: [B, T+I, D]
    out0_ptr,         # output for first T rows: [B, T, D]
    out1_ptr,         # output for next I rows: [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_seqlen_stride, Y_d_stride,
    out0_b_stride, out0_d_stride,
    out1_b_stride, out1_d_stride,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_stream = tl.program_id(1)  # 0 => first T rows, 1 => next I rows
    pid_d = tl.program_id(2)

    t_total = T + I

    # Compute row index for source
    src_row = pid_stream * t_total + pid_d  # when pid_stream==1, this is T+pid_d
    # Compute destination row indices
    dest_t = src_row if pid_stream == 0 else src_row - T

    d_offsets = tl.arange(0, BLOCK_D)

    # Loop over feature dimension in tiles
    for d_start in range(0, D, BLOCK_D):
        d = d_start + d_offsets
        mask = d < D
        src_ptr = Y_ptr + pid_b * Y_b_stride + src_row * Y_seqlen_stride + d * Y_d_stride
        if pid_stream == 0:
            dst_ptr = out0_ptr + pid_b * out0_b_stride + dest_t * out0_d_stride + d * out0_d_stride
        else:
            dst_ptr = out1_ptr + pid_b * out1_b_stride + (dest_t - T) * out1_d_stride + d * out1_d_stride
        vals = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Concatenate along sequence dimension in Triton.
        - Compute linear projection without bias in Triton: [B, T+I, D] @ [D, D] -> [B, T+I, D]
        - Split into encoder and image outputs in Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        # Ensure dtype is float32 for numerical consistency with the original code
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        # Make inputs contiguous
        E = encoder_hidden_states.contiguous()  # [B, T, D]
        H = hidden_states.contiguous()          # [B, I, D]
        W = process_weight.contiguous()         # [D, D]

        B, T, D = E.shape
        B2, I, D2 = H.shape
        assert B == B2 and D == D2, "Encoder and hidden states must have matching batch and hidden_dim."
        assert W.shape == (D, D), "process_weight must have shape [hidden_dim, hidden_dim]."
        M_total = T + I

        # 1) Concatenate encoder_hidden_states and hidden_states along sequence to [B, T+I, D]
        C = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)
        grid_concat = (B, 2, 1)  # stream 0: encoder rows, stream 1: hidden rows
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Compute Y = C @ W in Triton: Y is [B, M_total, D]
        Y = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)

        # Tiling parameters: choose moderate blocks; masks handle tail
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _matmul_batched_kernel[grid](
            C, W, Y,
            B, M_total, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split Y back into processed_encoder and processed_hidden in Triton
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        BLOCK_SPLIT = 64
        grid_split = (B, 2, triton.cdiv(D, BLOCK_SPLIT))  # stream 0: first T rows, stream 1: next I rows
        _split_copy_kernel[grid_split](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=BLOCK_SPLIT,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden