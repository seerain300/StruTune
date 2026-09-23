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
    # program ids
    pid_b = tl.program_id(0)  # batch
    pid_pos = tl.program_id(1)  # position in [T+I]

    t_total = T + I
    stream = pid_pos // t_total
    pos = pid_pos % t_total
    # map positions beyond T to hidden stream
    if stream == 1:
        pos = pos - T

    # compute source and destination pointers
    if stream == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # copy D elements
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)
        tl.store(dst_ptr + d * C_d_stride, val)


@triton.jit
def _split_kernel(
    src_ptr,      # [B, M_total, D]
    out0_ptr,     # [B, T, D]
    out1_ptr,     # [B, I, D]
    B, T, I, D,
    src_b_stride, src_m_stride, src_d_stride,
    out0_b_stride, out0_d_stride, out0_m_stride,
    out1_b_stride, out1_d_stride, out1_m_stride,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    if pid_m < T:
        src_row_ptr = src_ptr + pid_b * src_b_stride + pid_m * src_m_stride
        out0_row_ptr = out0_ptr + pid_b * out0_b_stride + pid_m * out0_m_stride
        for d in range(0, D):
            val = tl.load(src_row_ptr + d * src_d_stride)
            tl.store(out0_row_ptr + d * out0_d_stride, val)
    else:
        rel_m = pid_m - T
        src_row_ptr = src_ptr + pid_b * src_b_stride + rel_m * src_m_stride
        out1_row_ptr = out1_ptr + pid_b * out1_b_stride + rel_m * out1_m_stride
        for d in range(0, D):
            val = tl.load(src_row_ptr + d * src_d_stride)
            tl.store(out1_row_ptr + d * out1_d_stride, val)


def _triton_forward(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-optimized forward:
    - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton
    - Apply linear projection using torch.matmul (ensures numerical correctness)
    - Split results into encoder and hidden parts using Triton
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device for Triton kernels"

    B, T, D = encoder_hidden_states.shape
    Bi, I, Di = hidden_states.shape
    assert B == Bi, "Batch size must match for both inputs"
    assert D == Di, "Hidden dim must match for both inputs"
    assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

    # Ensure contiguous for simple stride arithmetic
    E = encoder_hidden_states.contiguous()
    H = hidden_states.contiguous()
    W = process_weight.contiguous()  # [D, D]

    # 1) Concatenate streams: [B, T+I, D] in Triton
    C_total = T + I
    C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

    grid_concat = (B, C_total)
    _concatenate_streams_kernel[grid_concat](
        E, H, C,
        B, T, I, D,
        E.stride(0), E.stride(1), E.stride(2),
        H.stride(0), H.stride(1), H.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        num_warps=4, num_stages=2,
    )

    # 2) Apply linear projection: C @ W^T -> [B, T+I, D]
    # Using torch.matmul to guarantee identical numerics to the reference.
    processed = torch.matmul(C, W.t())

    # 3) Split back into separate streams using Triton
    processed_encoder = torch.empty((B, T, D), device=processed.device, dtype=torch.float32)
    processed_hidden = torch.empty((B, I, D), device=processed.device, dtype=torch.float32)

    grid_split_encoder = (B, T)
    _split_kernel[grid_split_encoder](
        processed, processed_encoder, processed_hidden,
        B, T, I, D,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        num_warps=4, num_stages=2,
    )

    grid_split_hidden = (B, I)
    _split_kernel[grid_split_hidden](
        processed, processed_encoder, processed_hidden,
        B, T, I, D,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        num_warps=4, num_stages=2,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _triton_forward(hidden_states, encoder_hidden_states, process_weight)