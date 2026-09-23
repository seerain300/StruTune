import torch
import triton
import triton.language as tl

# Triton kernel: concatenate [B, T, D] and [B, I, D] into [B, T+I, D].
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
    # 2D launch grid: axis 0 over batch, axis 1 over total sequence rows (T+I)
    pid_b = tl.program_id(0)
    pid_row = tl.program_id(1)
    total = T + I

    # Determine source tensor/stream
    stream = pid_row // total  # 0 => encoder, 1 => hidden
    pos = pid_row % total      # row index in the concatenated sequence
    # Map pos to the corresponding index in the source sequence for hidden stream
    if stream == 1:
        pos = pos - T

    # Base pointers for source and destination
    if stream == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_row * C_seqlen_stride

    # Copy along feature dimension (D). We assume D is reasonably small for elementwise handling.
    # For robustness, we mask loads/stores; but since pid_row is bounded by total, and D by shape, this is safe.
    # If D is large, consider a loop over D with tl.arange and masked load/store.
    # Here we simply assume D fits in vectorized operations or is small enough to copy in chunks.
    # To keep it simple and correct, we copy 1 vector at a time using strides; Triton will handle scalar loop.
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)
        tl.store(dst_ptr + d * C_d_stride, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized concatenation, PyTorch matmul for projection, and split.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D tensors [B, seq_len, D]"
        assert process_weight.ndim == 2 and process_weight.shape[1] == hidden_states.shape[2], \
            "process_weight must be [D, D] where D == hidden_dim"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguity and correct dtype
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
            num_warps=4, num_stages=2,
        )

        # 2) Apply linear projection using PyTorch (no bias). This guarantees identical numerics to the reference.
        #    Original uses process_weight.T to match [D, D] weight.
        processed = torch.matmul(C, W.t())

        # 3) Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
