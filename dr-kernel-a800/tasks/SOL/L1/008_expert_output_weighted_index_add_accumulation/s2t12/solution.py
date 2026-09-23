import torch

# Try to import Triton. We'll provide a PyTorch fallback if unavailable.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def scatter_add_dim0_row_kernel(
        output_ptr,       # *bf16, shape [M, H]
        indices_ptr,      # *int32, shape [N]
        inputs_ptr,       # *bf16, shape [N, H] (expert_outputs)
        N,                # int32 (number of rows to process)
        stride_out_row: tl.constexpr,   # int64: elements between rows in output
        stride_out_col: tl.constexpr,   # int64: elements between cols in output (usually 1)
        stride_in_row: tl.constexpr,    # int64: elements between rows in inputs (usually H)
        stride_in_col: tl.constexpr,    # int64: elements between cols in inputs (usually 1)
        H: tl.constexpr,                # hidden_size (compile-time for vectorization)
    ):
        # One program per row i
        i = tl.program_id(axis=0)
        if i >= N:
            return

        # Load target row index (int32)
        idx = tl.load(indices_ptr + i)

        # Load the expert vector for row i (length H)
        cols = tl.arange(0, H)
        in_ptrs = inputs_ptr + i * stride_in_row + cols * stride_in_col
        vals = tl.load(in_ptrs)  # shape (H,), bfloat16

        # Compute output pointers for row idx and add vals across columns
        out_ptrs = output_ptr + idx * stride_out_row + cols * stride_out_col
        tl.atomic_add(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to preserve original semantics (random init), as in the reference
        output = final_hidden_states.clone()

        # If Triton/CUDA not available, fallback to PyTorch for correctness
        if (not _TRITON_AVAILABLE) or (not output.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Ensure inputs are contiguous
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Cast indices to int32 for efficient pointer arithmetic
        token_indices = token_indices.to(torch.int32).contiguous()

        # Shapes and strides
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Strides in elements
        stride_out_row = output.stride(0)  # typically H
        stride_out_col = output.stride(1)  # typically 1
        stride_in_row = expert_outputs.stride(0)  # typically H
        stride_in_col = expert_outputs.stride(1)  # typically 1

        # Launch Triton kernel: one program per row
        grid = (N,)

        scatter_add_dim0_row_kernel[grid](
            output, token_indices, expert_outputs,
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H,
            num_warps=4,  # reasonable default; can be tuned
        )

        return output


def run(*args):
    return ModelNew()(*args)
