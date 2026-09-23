import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add along dim=0 (row-wise)
# One program per row i: output[token_indices[i]] += expert_outputs[i]
@triton.jit
def scatter_add_dim0_row_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H] (expert_outputs)
    N,                # int32
    stride_out_row,   # int64 (elements between rows of output)
    stride_out_col,   # int64 (elements between cols of output, typically 1)
    stride_in_row,    # int64 (elements between rows of inputs, typically H)
    stride_in_col,    # int64 (elements between cols of inputs, typically 1)
    H: tl.constexpr,  # hidden_size (vector length)
):
    # Each program handles one row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + i)  # scalar int32

    # Load the expert vector for row i (length H), bfloat16
    in_ptrs = inputs_ptr + i * stride_in_row + tl.arange(0, H) * stride_in_col
    vals = tl.load(in_ptrs)  # shape (H,), bfloat16

    # Compute output pointers for row idx and add vals across columns
    out_ptrs = output_ptr + idx * stride_out_row + tl.arange(0, H) * stride_out_col
    # Atomically accumulate vals into output row
    tl.atomic_add(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # If Triton unavailable or tensors not on CUDA, we cannot run Triton.
        # The evaluation environment should provide CUDA; otherwise, this path is for safety.
        if (not TRITON_AVAILABLE) or (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            # Fallback: exact PyTorch behavior
            output = final_hidden_states.clone()
            # index_add along dim=0
            output.index_add_(dim=0, index=token_indices.to(torch.long), source=expert_outputs)
            return output

        # Ensure contiguity
        output = final_hidden_states.clone().contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Shapes and assertions
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Cast indices to int32 for Triton
        token_indices = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)  # typically H
        stride_in_col = expert_outputs.stride(1)  # typically 1

        # Launch Triton kernel: one program per row
        grid = (N,)

        scatter_add_dim0_row_kernel[grid](
            output, token_indices, expert_outputs,
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H=H,  # pass hidden_size as constexpr for vectorization
            num_warps=4,  # modest parallelism per program
        )

        return output


def run(*args):
    return ModelNew()(*args)
