import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states [B, T, D] and hidden_states [B, I, D]
# into output [B, T+I, D], placing encoder rows first, then hidden rows.
@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    C_ptr,        # concatenated output: [B, T+I, D]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seqlen_stride, C_d_stride,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_pos = tl.program_id(1)  # position in [T+I]

    # Map position to source stream: first T rows come from encoder, next I from hidden
    if pid_pos < T:
        src_ptr = E_ptr + pid_b * E_b_stride + pid_pos * E_t_stride
    else:
        src_pos = pid_pos - T
        src_ptr = H_ptr + pid_b * H_b_stride + src_pos * H_i_stride

    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # Vectorized copy over D dimension with masking
    for d in range(0, D):
        # We use BLOCK_D to create a masked load/store if needed; here D is exact, but we keep generality.
        val = tl.load(src_ptr + d * E_d_stride)  # E_d_stride == H_d_stride == C_d_stride == 1 for contiguous
        tl.store(dst_ptr + d * C_d_stride, val)


# Triton kernel: split the processed concatenated tensor Y [B, T+I, D] into two outputs:
# processed_encoder [B, T, D] (first T rows) and processed_hidden [B, I, D] (next I rows).
@triton.jit
def _split_seq_kernel(
    Y_ptr,        # [B, T+I, D]
    out0_ptr,     # [B, T, D]
    out1_ptr,     # [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_seq_stride, Y_d_stride,
    out0_b_stride, out0_t_stride, out0_d_stride,
    out1_b_stride, out1_i_stride, out1_d_stride,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # row index in [0, T)
    # For encoder output
    y_row_ptr = Y_ptr + pid_b * Y_b_stride + pid_t * Y_seq_stride
    out0_row_ptr = out0_ptr + pid_b * out0_b_stride + pid_t * out0_t_stride
    for d in range(0, D):
        val = tl.load(y_row_ptr + d * Y_d_stride)
        tl.store(out0_row_ptr + d * out0_d_stride, val)

    # For hidden output: rows T to T+I
    pid_i = tl.program_id(2)  # row index in [0, I)
    y_row_ptr2 = Y_ptr + pid_b * Y_b_stride + (pid_t + T) * Y_seq_stride
    out1_row_ptr = out1_ptr + pid_b * out1_b_stride + pid_i * out1_i_stride
    for d in range(0, D):
        val = tl.load(y_row_ptr2 + d * Y_d_stride)
        tl.store(out1_row_ptr + d * out1_d_stride, val)

    # Note: we launch with grid (B, T, I) so pid_t and pid_i are disjoint. We could fuse both into
    # a single kernel per row with a switch, but separating improves clarity and is fine.


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Uses torch.matmul for the linear projection (no bias): [B, T+I, D] @ process_weight.T
        - Splits the result back into two outputs in Triton.

        Returns:
            processed_encoder: [B, T, D]
            processed_hidden: [B, I, D]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D and process_weight.shape[0] == D and process_weight.shape[1] == D, "Hidden and weight dims must match."

        # Make tensors contiguous to simplify stride handling
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate streams in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_D=64,
            num_warps=4, num_stages=2,
        )

        # 2) Apply linear projection using PyTorch matmul for correctness: processed = C @ W.T
        processed = torch.matmul(C, W.t())

        # 3) Split back into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        grid_split = (B, T, I)
        _split_seq_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=64,
            num_warps=2, num_stages=2,
        )

        return processed_encoder, processed_hidden