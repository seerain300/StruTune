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
):
    # grid: (B, T+I)
    pid_b = tl.program_id(0)
    pid_pos = tl.program_id(1)
    t_total = T + I
    stream = pid_pos // t_total
    pos = pid_pos % t_total
    # Map to source row
    src_pos = pos if stream == 0 else (pos - T)
    # Compute pointers
    src_row_ptr = E_ptr + pid_b * E_b_stride + src_pos * E_t_stride
    dst_row_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_s_stride

    # Copy contiguous vector of length D
    for d in range(0, D):
        val = tl.load(src_row_ptr + d * E_d_stride)
        tl.store(dst_row_ptr + d * C_d_stride, val)


@triton.jit
def _batched_gemm_rowwise_kernel(
    X_ptr,        # input [B, M_total, D]
    W_ptr,        # weight [D, D]
    Y_ptr,        # output [B, M_total, D]
    B, M_total, D,
    X_b_stride, X_m_stride, X_d_stride,
    W0_stride, W1_stride,   # W strides: (row, col) -> W0_stride=col, W1_stride=row
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_N: tl.constexpr,  # tile over output columns
    BLOCK_K: tl.constexpr,  # tile over reduction dimension
):
    # Grid: (B*M_total, ceil_div(D, BLOCK_N))
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_row // M_total
    m = pid_row % M_total

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < D

    # Accumulator for this row's tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Reduction over K (input feature dim)
    for k0 in range(0, D, BLOCK_K):
        off_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = off_k < D

        # Load X[b, m, k] as vector
        x_row_ptr = X_ptr + b * X_b_stride + m * X_m_stride
        x_vals = tl.load(x_row_ptr + off_k * X_d_stride, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load W[k, n] as BLOCK_K x BLOCK_N
        w_tile = tl.load(
            W_ptr + off_k[:, None] * W0_stride + off_n[None, :] * W1_stride,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc[n] += sum_k x_vals[k] * w_tile[k, n]
        acc += tl.sum(w_tile * x_vals[:, None], axis=0)

    # Store result to Y[b, m, :]
    y_row_ptr = Y_ptr + b * Y_b_stride + m * Y_m_stride
    tl.store(y_row_ptr + off_n * Y_d_stride, acc, mask=mask_n)


@triton.jit
def _split_copy_rows_kernel(
    Y_ptr,            # [B, T+I, D] processed tensor
    out0_ptr,         # [B, T, D] destination for first T rows
    out1_ptr,         # [B, I, D] destination for next I rows
    B, T, I, D,
    Y_b_stride, Y_m_stride, Y_d_stride,
    out0_b_stride, out0_d_stride,
    out1_b_stride, out1_d_stride,
    BLOCK_D: tl.constexpr,
):
    # grid: (B, 2, ceil_div(D, BLOCK_D)) where stream=0 -> first T rows, stream=1 -> next I rows
    pid_b = tl.program_id(0)
    pid_stream = tl.program_id(1)
    pid_dn = tl.program_id(2)

    # Compute destination rows
    if pid_stream == 0:
        m = pid_dn  # first T rows
    else:
        m = pid_dn + T  # next I rows

    if m >= (T + I):
        return  # handled by mask

    off_d = pid_dn * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = off_d < D

    src_row_ptr = Y_ptr + pid_b * Y_b_stride + m * Y_m_stride
    dst0_row_ptr = out0_ptr + pid_b * out0_b_stride + m * out0_d_stride  # valid for m < T
    dst1_row_ptr = out1_ptr + pid_b * out1_b_stride + (m - T) * out1_d_stride  # valid for m >= T

    # Copy contiguous vector of length D
    for d in range(0, D):
        val = tl.load(src_row_ptr + d * Y_d_stride)
        if m < T:
            tl.store(dst0_row_ptr + d * out0_d_stride, val)
        else:
            tl.store(dst1_row_ptr + d * out1_d_stride, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, D]
          processed = concatenated @ process_weight.T                       # [B, T+I, D]
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, seq_len, D]"
        assert process_weight.dim() == 2 and process_weight.shape[1] == process_weight.shape[0], "Weight must be square [D, D]"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D, "encoder_hidden_states must have same hidden_dim D"
        assert process_weight.shape[1] == D, "process_weight second dim must equal hidden_dim D"

        # Ensure contiguous tensors
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        device = E.device
        dtype = torch.float32

        # 1) Concatenate in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=device, dtype=dtype)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = C @ W in Triton (no bias). Output Y: [B, T+I, D]
        Y = torch.empty((B, C_total, D), device=device, dtype=dtype)

        # Launch batched GEMM row-wise: grid over (B*C_total, tiles over D)
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B * C_total, triton.cdiv(D, BLOCK_N))
        _batched_gemm_rowwise_kernel[grid_gemm](
            C, W, Y,
            B, C_total, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),  # W is [D, D], so (row stride=W.stride(1), col stride=W.stride(0))
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split processed into encoder and hidden streams in Triton
        processed_encoder = torch.empty((B, T, D), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=dtype)

        BLOCK_SPLIT = 64
        grid_split = (B, 2, triton.cdiv(D, BLOCK_SPLIT))
        _split_copy_rows_kernel[grid_split](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=BLOCK_SPLIT,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
