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
    # 3D grid over (batch, encoder rows, hidden rows)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)

    total = T + I
    # Derive source and position
    stream = pid_i // total  # 0 => encoder, 1 => hidden
    pos = pid_i % total
    if stream == 1:
        pos = pos - T  # map to hidden indices

    # Compute base pointers
    src_base = E_ptr + pid_b * E_b_stride if stream == 0 else H_ptr + pid_b * H_b_stride
    dst_base = C_ptr + pid_b * C_b_stride

    # Copy D-length vector
    for d in range(0, D):
        val = tl.load(src_base + (pos * E_t_stride if stream == 0 else pid_t * H_i_stride) + d * (E_d_stride if stream == 0 else H_d_stride))
        tl.store(dst_base + pos * C_seqlen_stride + d * C_d_stride, val)


@triton.jit
def _row_matmul_kernel(
    X_ptr,        # input: [B, M_total, D], here M_total = T + I
    W_ptr,        # process_weight: [D, D]
    Y_ptr,        # output: [B, M_total, D]
    B, M_total, D,
    X_b_stride, X_m_stride, X_d_stride,
    W0_stride, W1_stride,
    Y_b_stride, Y_m_stride, Y_d_stride,
):
    # 2D grid: axis0 over B*M_total rows, axis1 over output feature tiles
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute batch and row index
    b = pid_row // M_total
    m = pid_row % M_total

    # Output feature tile
    n_start = pid_n * BLOCK_N
    out_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over input feature dimension K = D
    for k in range(0, D):
        x_val = tl.load(X_ptr + b * X_b_stride + m * X_m_stride + k * X_d_stride)
        w_vec = tl.load(W_ptr + k * W0_stride + n_start + tl.arange(0, BLOCK_N) * W1_stride)
        out_vec += x_val * w_vec

    # Store results
    tl.store(Y_ptr + b * Y_b_stride + m * Y_m_stride + (n_start + tl.arange(0, BLOCK_N)) * Y_d_stride, out_vec, mask=(n_start + tl.arange(0, BLOCK_N)) < D)


@triton.jit
def _split_rows_kernel(
    Y_ptr,            # input processed: [B, T+I, D]
    out0_ptr,         # output for encoder: [B, T, D]
    out1_ptr,         # output for hidden: [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_m_stride, Y_d_stride,
    out0_b_stride, out0_d_stride,
    out1_b_stride, out1_d_stride,
):
    # Grid over (batch, stream rows)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    stream = pid_m // I  # 0 => encoder rows [0..T-1], 1 => hidden rows [T..T+I-1]
    row = pid_m % I
    if stream == 0:
        src_row = row  # 0..T-1
        dst_row = row  # 0..T-1
        out = out0_ptr
        out_b_stride, out_d_stride = out0_b_stride, out0_d_stride
    else:
        src_row = row + T  # T..T+I-1
        dst_row = row       # 0..I-1
        out = out1_ptr
        out_b_stride, out_d_stride = out1_b_stride, out1_d_stride

    for d in range(0, D):
        val = tl.load(Y_ptr + pid_b * Y_b_stride + src_row * Y_m_stride + d * Y_d_stride)
        tl.store(out + pid_b * (out_b_stride) + dst_row * (Y_m_stride) + d * (Y_d_stride), val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of:
        concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, D]
        processed = concatenated @ process_weight.T                      # [B, T+I, D]
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, length, D]"
        assert process_weight.dim() == 2 and process_weight.shape[0] == process_weight.shape[1], "process_weight must be square [D, D]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        M_total = T + I

        # Ensure inputs are on CUDA and contiguous
        device = hidden_states.device
        dtype = hidden_states.dtype

        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()  # [D, D]
        # Allocate concatenated buffer [B, T+I, D]
        C = torch.empty((B, M_total, D), device=device, dtype=torch.float32)
        # Launch concatenation kernel: grid over (B, T, I)
        grid_concat = (B, T, I)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # Allocate output buffer [B, M_total, D]
        Y = torch.empty((B, M_total, D), device=device, dtype=torch.float32)

        # Launch row-wise matmul kernel: grid over (B*M_total rows, feature tiles)
        BLOCK_N = 64
        grid_matmul = (B * M_total, triton.cdiv(D, BLOCK_N))
        _row_matmul_kernel[grid_matmul](
            C, W,
            Y,
            B, M_total, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs and split via Triton
        processed_encoder = torch.empty((B, T, D), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=torch.float32)

        # Launch split kernel: grid over (B, T+I rows)
        grid_split = (B, M_total)
        _split_rows_kernel[grid_split](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        # If original dtype is not float32, cast outputs back to match inputs
        if dtype != torch.float32:
            processed_encoder = processed_encoder.to(dtype)
            processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
