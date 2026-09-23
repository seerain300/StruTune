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
    C_b_stride, C_t_stride, C_d_stride,
):
    # Grid: (B, T+I)
    pid_b = tl.program_id(0)
    pid_pos = tl.program_id(1)

    t_total = T + I
    stream = pid_pos // t_total
    pos = pid_pos % t_total
    # Map positions: first T rows from encoder, next I from hidden
    src_stream = 0 if stream == 0 else 1
    src_pos = pos - T if stream == 1 else pos

    # Compute base pointers
    if src_stream == 0:
        src_row_ptr = E_ptr + pid_b * E_b_stride + src_pos * E_t_stride
    else:
        src_row_ptr = H_ptr + pid_b * H_b_stride + src_pos * H_i_stride

    dst_row_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_t_stride

    # Copy the D features (contiguous)
    for d in range(0, D):
        val = tl.load(src_row_ptr + d * E_d_stride)
        tl.store(dst_row_ptr + d * C_d_stride, val)


@triton.jit
def _split_seq_kernel(
    C_ptr,        # concatenated: [B, T+I, D]
    out0_ptr,     # processed_encoder: [B, T, D]
    out1_ptr,     # processed_hidden: [B, I, D]
    B, T, I, D,
    C_b_stride, C_t_stride, C_d_stride,
    out0_b_stride, out0_d_stride,
    out1_b_stride, out1_d_stride,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, T) for out0, (B, I) for out1
    pid_b = tl.program_id(0)
    pid_idx = tl.program_id(1)

    # For out0: copy rows 0..T-1
    row = pid_idx  # within [0, T)
    # Pointers
    src_ptr = C_ptr + pid_b * C_b_stride + row * C_t_stride
    dst_ptr = out0_ptr + pid_b * out0_b_stride + row * out0_d_stride

    # For out1: copy rows T..T+I-1
    # We compute this with a separate program id over (B, I)
    # The calling code uses two separate launches for out0 and out1

    # Copy in tiles along D
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        vals = tl.load(src_ptr + offs * C_d_stride, mask=mask, other=0.0)
        tl.store(dst_ptr + offs * out0_d_stride, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure contiguity for predictable strides
        E = encoder_hidden_states.contiguous()  # [B, T, D]
        H = hidden_states.contiguous()          # [B, I, D]
        Wt = process_weight.t().contiguous()    # [D, D], process_weight is [D, D] in the original code

        B = E.shape[0]
        T = E.shape[1]
        I = H.shape[1]
        D = E.shape[2]
        assert Wt.shape[0] == D and Wt.shape[1] == D, "process_weight must be [hidden_dim, hidden_dim]"

        # 1) Concatenate in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=E.dtype)
        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Compute processed = C @ Wt using PyTorch (highly optimized and numerically robust)
        processed = torch.matmul(C, Wt)

        # 3) Split back in Triton
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=E.dtype)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=E.dtype)

        # Launch for encoder rows [0, T)
        grid_encoder = (B, T)
        _split_seq_kernel[grid_encoder](
            processed, processed_encoder, processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=128, num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden