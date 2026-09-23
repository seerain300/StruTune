import torch

# Try to import Triton; if unavailable, we will rely on PyTorch fallback.
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add along dim=0 (row-wise).
# Grid: one program per row i in [0, N).
@triton.jit
def scatter_add_dim0_row_kernel(
    output_ptr,       # *bf16, shape [M, H]
    indices_ptr,      # *int32, shape [N]
    inputs_ptr,       # *bf16, shape [N, H] (expert_outputs)
    N,                # int32
    stride_out_row,   # int64: elements between consecutive rows in output
    stride_out_col,   # int64: elements between consecutive cols in output (typically 1)
    stride_in_row,    # int64: elements between consecutive rows in inputs (typically H)
    stride_in_col,    # int64: elements between consecutive cols in inputs (typically 1)
    H: tl.constexpr,  # hidden_size, compile-time constant for vectorization
):
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index for this expert output
    idx = tl.load(indices_ptr + i)  # int32

    # Load the expert vector (length H) for row i
    in_ptrs = inputs_ptr + i * stride_in_row + tl.arange(0, H) * stride_in_col
    vals = tl.load(in_ptrs)  # bfloat16 vector of length H

    # Compute output pointers for row idx and atomic add across columns
    out_ptrs = output_ptr + idx * stride_out_row + tl.arange(0, H) * stride_out_col
    tl.atomic_add(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure we run the Triton kernel as required; fall back only if Triton unavailable or tensors not on CUDA.
        use_triton = (
            TRITON_AVAILABLE
            and final_hidden_states.is_cuda
            and expert_outputs.is_cuda
            and token_indices.is_cuda
        )
        if not use_triton:
            # Preserve original semantics
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Triton path: ensure contiguity and dtype
        output = final_hidden_states.clone().contiguous()          # preserve random init
        expert_outputs = expert_outputs.contiguous()
        token_indices_i32 = token_indices.to(torch.int32).contiguous()

        # Shapes and assertions
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices_i32.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)
        stride_in_col = expert_outputs.stride(1)

        # Launch Triton kernel: one program per row
        grid = (N,)
        scatter_add_dim0_row_kernel[grid](
            output, token_indices_i32, expert_outputs,
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H=H,
            num_warps=4,
        )
        return output


def run(*args):
    return ModelNew()(*args)
