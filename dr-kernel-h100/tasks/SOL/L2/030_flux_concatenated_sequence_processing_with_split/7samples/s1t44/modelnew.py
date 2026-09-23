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
    pid_b = tl.program_id(0)  # batch dimension
    pid_pos = tl.program_id(1)  # position in [T+I]
    total = T + I
    stream = pid_pos // total  # 0 => encoder, 1 => hidden
    pos = pid_pos % total
    if stream == 1:
        pos = pos - T

    src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride if stream == 0 else H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # Copy vector of length D (assume D is reasonably small; if large, loop in tiles)
    # We implement a simple per-dimension copy guarded by mask (not strictly necessary if D is constexpr).
    for d in range(0, 64):  # D is dynamic; for correctness we handle typical D <= 1024 by chunking or rely on host-side D
        # The above simplistic loop is not ideal; instead, we rely on host-side D and a while loop.
        pass

    # Better approach: while loop over D dimension with masking
    d = 0
    while d < D:
        val = tl.load(src_ptr + d * E_d_stride if stream == 0 else src_ptr + d * H_d_stride, mask=d < D, other=0.0)
        tl.store(dst_ptr + d * C_d_stride, val, mask=d < D)
        d += 1


@triton.jit
def _split_copy_rows_kernel(
    Y_ptr,                # processed: [B, T+I, D]
    out0_ptr, out1_ptr,   # outputs: [B, T, D] and [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_t_stride, Y_d_stride,
    out0_b_stride, out0_d_stride,
    out1_b_stride, out1_d_stride,
    BLOCK_D: tl.constexpr,
):
    # Grid dims: (B, T, ceil_div(D, BLOCK_D)) for out0; (B, I, ceil_div(D, BLOCK_D)) for out1
    pid_b = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_tile = tl.program_id(2)

    d_offsets = pid_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # Copy rows from Y to out0
    src_row = pid_row  # encoder rows are 0..T-1
    src_ptr = Y_ptr + pid_b * Y_b_stride + src_row * Y_t_stride + d_offsets * Y_d_stride
    dst_ptr0 = out0_ptr + pid_b * out0_b_stride + src_row * out0_d_stride + d_offsets * out0_d_stride
    tl.store(dst_ptr0, tl.load(src_ptr, mask=mask_d, other=0.0), mask=mask_d)

    # Copy rows from Y to out1
    src_row2 = pid_row + T  # hidden rows are T..T+I-1
    src_ptr2 = Y_ptr + pid_b * Y_b_stride + src_row2 * Y_t_stride + d_offsets * Y_d_stride
    dst_ptr1 = out1_ptr + pid_b * out1_b_stride + (pid_row) * out1_d_stride + d_offsets * out1_d_stride
    tl.store(dst_ptr1, tl.load(src_ptr2, mask=mask_d, other=0.0), mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-enabled version:
        1) Concatenate [B, T, D] and [B, I, D] along sequence dim using Triton.
        2) Apply linear projection via torch.matmul (C @ W^T).
        3) Split results back into [B, T, D] and [B, I, D] using Triton copy kernels.
        """
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()  # [D, D]
        B, T, D = E.shape
        B2, I, D2 = H.shape
        assert B == B2 and D == D2, "Mismatched shapes for encoder_hidden_states and hidden_states"

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
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection using torch for numerical correctness
        # Y = C @ W.T
        Wt = W.t().contiguous()  # [D, D]
        Y = torch.matmul(C, Wt)  # [B, T+I, D]

        # 3) Split using Triton
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        BLOCK_D = 128
        grid_encoder = (B, T, (D + BLOCK_D - 1) // BLOCK_D)
        grid_hidden = (B, I, (D + BLOCK_D - 1) // BLOCK_D)

        _split_copy_rows_kernel[grid_encoder](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden