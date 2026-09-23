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
    pid_t = tl.program_id(1)  # position in [0, T)
    pid_i = tl.program_id(2)  # position in [0, I)

    total = T + I
    # The third grid dimension is either T or I; it encodes which input to copy
    src_stream = pid_i  # 0 => encoder rows, 1 => hidden rows
    pos = pid_t + (src_stream * T)  # pos in [0, T) if src_stream=0, else pos in [T, T+I)

    # Compute source and destination pointers
    if src_stream == 0:
        src_row = pid_b * E_b_stride + pid_t * E_t_stride
        dst_row = pid_b * C_b_stride + pos * C_seqlen_stride
    else:
        src_row = pid_b * H_b_stride + (pid_t) * H_i_stride  # pid_t here is in [0, I)
        dst_row = pid_b * C_b_stride + (pos) * C_seqlen_stride  # pos = T + pid_t

    # Copy vector of length D
    # D is assumed to be the last dimension; strides reflect that
    # Masking is not strictly needed since pid_t and pid_i are within bounds by grid definition.
    for d in range(0, D):
        val = tl.load(E_ptr + src_row + d * E_d_stride) if src_stream == 0 else tl.load(H_ptr + src_row + d * H_d_stride)
        tl.store(C_ptr + dst_row + d * C_d_stride, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure contiguous tensors
        E = encoder_hidden_states.contiguous()  # [B, T, D]
        H = hidden_states.contiguous()         # [B, I, D]
        W = process_weight.contiguous()        # [D, D]

        B = E.shape[0]
        T = E.shape[1]
        I = H.shape[1]
        D = E.shape[2]

        # 1) Concatenate streams using Triton: output [B, T+I, D]
        total = T + I
        C = torch.empty((B, total, D), device=E.device, dtype=E.dtype)

        # Launch grid: (B, T, I). Each program copies one row from either E or H into C.
        grid_concat = (B, T, I)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Apply linear projection using PyTorch (highly optimized and numerically robust)
        # processed = C @ W.T -> [B, T+I, D]
        W_t = W.transpose(0, 1).contiguous()  # [D, D]
        processed = torch.matmul(C, W_t)

        # 3) Split back into encoder and image streams using torch slicing
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden