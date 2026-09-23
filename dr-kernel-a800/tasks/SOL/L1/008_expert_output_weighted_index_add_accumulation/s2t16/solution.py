import torch
import triton
import triton.language as tl


@triton.jit
def row_scatter_add_dim0_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H] (expert_outputs)
    N,                # int32, number of rows to process
    stride_out_row,   # int64, elements between rows of output
    stride_out_col,   # int64, elements between cols of output (typically 1)
    stride_in_row,    # int64, elements between rows of inputs (typically H)
    stride_in_col,    # int64, elements between cols of inputs (typically 1)
    H: tl.constexpr,  # hidden_size, compile-time for vectorization
):
    # One program per row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + i)  # scalar int32

    # Load the expert vector for row i (length H), dtype bfloat16
    in_ptrs = inputs_ptr + i * stride_in_row + tl.arange(0, H) * stride_in_col
    vals = tl.load(in_ptrs)  # shape (H,), bfloat16

    # Compute output pointers for row idx and atomically add vals across columns
    out_ptrs = output_ptr + idx * stride_out_row + tl.arange(0, H) * stride_out_col
    tl.atomic_add(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Fallback to PyTorch if not on CUDA
        if (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Preserve original random initialization by cloning
        output = final_hidden_states.clone().contiguous()

        # Shapes and assertions
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        # Cast indices to int32 for pointer arithmetic
        token_indices = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)  # typically H
        stride_in_col = expert_outputs.stride(1)  # typically 1

        # Launch Triton kernel: one program per row
        grid = (N,)

        row_scatter_add_dim0_kernel[grid](
            output, token_indices, expert_outputs,
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H=H,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
