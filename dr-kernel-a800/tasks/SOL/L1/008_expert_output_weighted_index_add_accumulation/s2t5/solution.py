import torch
import triton
import triton.language as tl


@triton.jit
def scatter_store_dim0_row_kernel(
    output_ptr,        # *bf16, shape [M, H]
    indices_ptr,       # *int32, shape [N]
    inputs_ptr,        # *bf16, shape [N, H] (expert_outputs)
    N,                 # int32
    H,                 # int32
    stride_out_row,    # int64, elements between rows of output
    stride_out_col,    # int64, elements between cols of output (usually 1)
    stride_in_row,     # int64, elements between rows of inputs (usually H)
    stride_in_col,     # int64, elements between cols of inputs (usually 1)
):
    # One program per row i
    i = tl.program_id(axis=0)
    if i >= N:
        return

    # Load target row index (int32 scalar)
    idx = tl.load(indices_ptr + i)

    # Load the expert vector for row i: shape (H,) bfloat16
    cols = tl.arange(0, H)
    in_ptrs = inputs_ptr + i * stride_in_row + cols * stride_in_col
    vals = tl.load(in_ptrs)  # vector of length H

    # Compute output pointers for row idx and store vals across columns
    out_ptrs = output_ptr + idx * stride_out_row + cols * stride_out_col
    tl.store(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Use Triton on CUDA devices; fallback to PyTorch on non-CUDA
        if (final_hidden_states.device.type != "cuda" or
            expert_outputs.device.type != "cuda" or
            token_indices.device.type != "cuda"):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Preserve original semantics: clone the accumulator (random init)
        output = final_hidden_states.clone().contiguous()

        # Shapes
        M, H = output.shape
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Ensure inputs are contiguous
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)
        stride_in_col = expert_outputs.stride(1)

        # Launch Triton kernel: one program per row
        grid = (N,)

        scatter_store_dim0_row_kernel[grid](
            output, token_indices, expert_outputs,
            N, H,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            num_warps=4,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
