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
    t_total = T + I
    stream = pid_pos // t_total  # 0 => encoder, 1 => hidden
    pos = pid_pos % t_total
    if stream == 1:
        pos = pos - T  # map to hidden index when stream == 1

    # Compute pointers
    src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride if stream == 0 else H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # Copy vector of length D
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)
        tl.store(dst_ptr + d * C_d_stride, val)


@triton.jit
def _split_copy_kernel(
    SRC_ptr,  # source tensor: [B, T+I, D]
    OUT0_ptr, # output 0: [B, T, D]
    OUT1_ptr, # output 1: [B, I, D]
    B, T, I, D,
    SRC_b_stride, SRC_seq_stride, SRC_d_stride,
    OUT0_b_stride, OUT0_seq_stride, OUT0_d_stride,
    OUT1_b_stride, OUT1_seq_stride, OUT1_d_stride,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_seq = tl.program_id(1)  # 0..T-1 for OUT0, T..T+I-1 for OUT1
    pid_tile = tl.program_id(2)  # tile over D

    # Compute d offsets for this tile
    d_offsets = pid_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Determine which output (encoder vs hidden)
    m_total = T + I
    stream = pid_seq // I  # 0 => encoder, 1 => hidden
    pos = pid_seq % m_total
    if stream == 1:
        pos = pos - T

    # Load from SRC
    src_row_ptr = SRC_ptr + pid_b * SRC_b_stride + pos * SRC_seq_stride
    vals = tl.load(src_row_ptr + d_offsets * SRC_d_stride, mask=d_mask, other=0.0)

    # Store to corresponding output
    out0_row_ptr = OUT0_ptr + pid_b * OUT0_b_stride + pos * OUT0_seq_stride
    out1_row_ptr = OUT1_ptr + pid_b * OUT1_b_stride + (pos - T) * OUT1_seq_stride  # pos-T for hidden rows

    if stream == 0:
        tl.store(out0_row_ptr + d_offsets * OUT0_d_stride, vals, mask=d_mask)
    else:
        tl.store(out1_row_ptr + d_offsets * OUT1_d_stride, vals, mask=d_mask)

# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [batch, img_seq_len, hidden_dim] (float32)
        encoder_hidden_states: [batch, text_seq_len, hidden_dim] (float32)
        process_weight: [hidden_dim, hidden_dim] (float32), used as projection W for matmul
        Returns: (processed_encoder: [batch, text_seq_len, hidden_dim], processed_hidden: [batch, img_seq_len, hidden_dim])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        # Ensure dtype is float32 for correctness
        dtype = torch.float32
        E = encoder_hidden_states.contiguous().to(dtype)
        H = hidden_states.contiguous().to(dtype)
        W = process_weight.contiguous().to(dtype)  # [D, hidden_dim] (here hidden_dim == D)

        B = E.shape[0]
        T = E.shape[1]
        I = H.shape[1]
        D = E.shape[2]
        assert W.shape[0] == D and W.shape[1] == D, "process_weight must be [D, D]"

        # 1) Concatenate streams in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=dtype)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Matmul using PyTorch (correct and robust): processed = C @ W
        # C: [B, T+I, D], W: [D, D] -> processed: [B, T+I, D]
        processed = torch.matmul(C, W)  # no bias

        # 3) Split using Triton kernel
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=dtype)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=dtype)

        BLOCK_D = 128  # tile along feature dimension
        grid_split = (B, T + I, triton.cdiv(D, BLOCK_D))
        _split_copy_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
