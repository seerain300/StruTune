import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel_1d(
    e_ptr,                # *dtype, encoder_hidden_states: [B, T, H]
    i_ptr,                # *dtype, hidden_states: [B, I, H]
    out_ptr,              # *dtype, output X_cat: [B, T+I, H]
    B, T, I,              # int32 sizes
    stride_eb, stride_et, stride_eh,   # strides for encoder_hidden_states
    stride_ib, stride_ih,               # strides for hidden_states
    stride_ob, stride_om, stride_oh,   # strides for output
    H: tl.constexpr,                 # hidden dimension, compile-time constant for loop unrolling
):
    # 1D grid: one program per (batch, row)
    pid = tl.program_id(0)
    rows_per_batch = T + I
    b = pid // rows_per_batch
    p = pid % rows_per_batch

    # Determine source tensor: use encoder for p < T, else image
    use_encoder = p < T

    # Compute base pointers for the current row
    if use_encoder:
        row_ptr = e_ptr + b * stride_eb + p * stride_et
    else:
        row_ptr = i_ptr + b * stride_ib + (p - T) * stride_ih

    # Output row pointer
    out_row_ptr = out_ptr + b * stride_ob + p * stride_om

    # Copy H elements from source to output row (H is tl.constexpr, loop is unrolled)
    for j in range(0, H):
        val = tl.load(row_ptr + j * (stride_eh if use_encoder else stride_ih))
        tl.store(out_row_ptr + j * stride_oh, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Allocate output concatenated matrix [B, T+I, H]
        X_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch Triton kernel to build concatenated matrix
        grid = (B * (T + I),)  # one program per (batch, row)
        cat_rows_kernel_1d[grid](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I,
            # strides
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *X_cat.stride(),
            H=H,                # pass H as constexpr for loop unrolling
            num_warps=1, num_stages=1,
        )

        # Linear projection: Y = X_cat @ process_weight  (process_weight: [H, H], no bias)
        processed = torch.matmul(X_cat, process_weight)

        # Split back into encoder and image streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
