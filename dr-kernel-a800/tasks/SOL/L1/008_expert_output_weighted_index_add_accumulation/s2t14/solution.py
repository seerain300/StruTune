import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_row_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H] (expert_outputs)
    N,                # int32 (number of rows in inputs)
    stride_out_row,   # int64 (elements between rows of output)
    stride_out_col,   # int64 (elements between cols of output)
    stride_in_row,    # int64 (elements between rows of inputs)
    stride_in_col,    # int64 (elements between cols of inputs)
    H: tl.constexpr,  # hidden_size (compile-time for vectorization)
):
    # Each program handles one row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + i)  # scalar int32

    # Load expert vector for row i: shape (H,)
    in_ptrs = inputs_ptr + i * stride_in_row + tl.arange(0, H) * stride_in_col
    vals = tl.load(in_ptrs)  # bfloat16 vector of length H

    # Compute output pointers for row idx and add vals across columns
    out_ptrs = output_ptr + idx * stride_out_row + tl.arange(0, H) * stride_out_col
    # Atomic add vector vals into output row
    tl.atomic_add(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # If Triton/CUDA not available, fall back to PyTorch for correctness
        use_triton = (
            final_hidden_states.is_cuda
            and expert_outputs.is_cuda
            and token_indices.is_cuda
        )

        # Clone the accumulator to preserve original semantics (random init)
        output = final_hidden_states.clone()

        if not use_triton:
            # PyTorch fallback: exact semantics
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Ensure tensors are contiguous and dtypes are correct
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Cast indices to int32 for pointer arithmetic
        token_indices = token_indices.to(torch.int32).contiguous()

        # Shapes and strides (in elements)
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)
        stride_in_col = expert_outputs.stride(1)

        # Launch Triton kernel: one program per row
        grid = (N,)
        scatter_add_dim0_row_kernel[grid](
            output, token_indices, expert_outputs,
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H,
            num_warps=4,  # tune for performance
            num_stages=1,
        )
        return output


def run(*args):
    return ModelNew()(*args)
