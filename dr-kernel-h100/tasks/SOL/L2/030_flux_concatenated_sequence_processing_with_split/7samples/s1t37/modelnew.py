import torch
import triton
import triton.language as tl

# 1) Triton kernel for concatenation: C = [B, T+I, D] = [encoder] || [hidden]
@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    C_ptr,        # concatenated output: [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seq_stride, C_d_stride,
):
    pid_b = tl.program_id(0)  # batch
    pid_pos = tl.program_id(1)  # position in [0, T+I)
    stream = pid_pos // (T + I)  # 0 => encoder, 1 => hidden
    pos = pid_pos % (T + I)
    if stream == 1:
        pos = pos - T

    src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seq_stride

    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)
        tl.store(dst_ptr + d * C_d_stride, val)


# 2) Triton kernel for split (copy rows from processed to two outputs)
# We'll use this to write processed_encoder = processed[:, :T, :] and
# processed_hidden = processed[:, T:, :].
@triton.jit
def _split_copy_kernel(
    Y_ptr,        # processed: [B, T+I, D]
    Y0_ptr,       # output for encoder stream: [B, T, D]
    Y1_ptr,       # output for image stream: [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_seq_stride, Y_d_stride,
    Y0_b_stride, Y0_d_stride,
    Y1_b_stride, Y1_d_stride,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_stream = tl.program_id(1)  # 0 => encoder, 1 => hidden
    pid_row = tl.program_id(2)

    # Determine source row in Y based on stream
    if pid_stream == 0:
        src_row = pid_row
    else:
        src_row = pid_row + T

    src_ptr = Y_ptr + pid_b * Y_b_stride + src_row * Y_seq_stride
    if pid_stream == 0:
        dst_ptr = Y0_ptr + pid_b * Y0_b_stride + pid_row * Y0_d_stride
    else:
        dst_ptr = Y1_ptr + pid_b * Y1_b_stride + pid_row * Y1_d_stride

    # Copy D features
    for d in range(0, BLOCK_D):
        if d < D:
            val = tl.load(src_ptr + d * Y_d_stride)
            tl.store(dst_ptr + d * (Y0_d_stride if pid_stream == 0 else Y1_d_stride), val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton version:
        - Concatenate [B, T, D] and [B, I, D] along sequence into [B, T+I, D] (Triton).
        - Compute processed = concatenated @ process_weight.T using torch.matmul for correctness.
        - Split processed into [B, T, D] and [B, I, D] (Triton copy).
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, S, D] tensors."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        # Ensure all tensors are on the same device and dtype
        assert encoder_hidden_states.device == hidden_states.device, "Encoder and hidden tensors must be on the same device."
        device = hidden_states.device
        # We will compute in float32 for stability; original code uses float32 by default
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()  # process_weight is [D, D]

        # 1) Concatenate in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=device, dtype=torch.float32)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Matmul in PyTorch to guarantee numerical correctness:
        #   processed = C @ W^T, where W is [D, D]
        processed = torch.matmul(C, W.t())

        # 3) Split in Triton: copy rows into two outputs
        processed_encoder = torch.empty((B, T, D), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=torch.float32)

        grid_split = (B, 2, T)  # first B, 2 streams, and T rows for encoder
        _split_copy_kernel[grid_split](
            processed,
            processed_encoder,
            processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=64,  # vectorized copy over D, mask handles remainder
            num_warps=4, num_stages=2,
        )

        # For rows beyond T, copy into hidden output
        grid_split_hidden = (B, 2, I)
        _split_copy_kernel[grid_split_hidden](
            processed,
            processed_encoder,
            processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden