import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_elementwise_kernel(
    output_ptr,        # *bf16, shape [M, H]
    indices_ptr,       # *int32, shape [N]
    inputs_ptr,        # *bf16, shape [N, H] (expert_outputs)
    M,                 # int32: number of rows in output (batch_size * seq_len)
    N,                 # int32: number of rows in expert_outputs (batch_size * seq_len * num_experts_per_tok)
    H,                 # int32: hidden_size
    stride_out_row,    # int64: output stride along rows
    stride_out_col,    # int64: output stride along cols
    stride_in_row,     # int64: inputs stride along rows
    stride_in_col,     # int64: inputs stride along cols
):
    # 2D grid over (row_index in [0, M), column j in [0, H))
    row_index = tl.program_id(axis=0)  # 0..M-1
    j = tl.program_id(axis=1)          # 0..H-1

    # Defensive guard (grid should be exactly (M, H))
    if row_index >= M:
        return

    # Accumulate contributions from all i where token_indices[i] == row_index.
    # This avoids atomics and ensures correctness even with duplicate indices.
    for i in range(0, N):
        idx = tl.load(indices_ptr + i)   # int32
        # Load the j-th column of expert_outputs for row i (scalar bfloat16)
        val = tl.load(inputs_ptr + i * stride_in_row + j * stride_in_col)
        # If this i maps to the current row_index, accumulate
        if idx == row_index:
            out_ptr = output_ptr + row_index * stride_out_row + j * stride_out_col
            tl.atomic_add(out_ptr, val)  # atomic add on a single element


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton-only computation: clone to preserve original semantics (random init)
        output = final_hidden_states.clone()

        # Ensure tensors are on CUDA and contiguous
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton execution."
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.to(torch.int32).contiguous()

        # Shapes
        M, H = output.shape  # output is (M, H), M = batch_size * seq_len
        N = expert_outputs.shape[0]
        assert expert_outputs.shape == (N, H), "expert_outputs must have shape (num_selected_tokens, hidden_size)"
        assert token_indices.shape[0] == N, "token_indices length must match expert_outputs rows"

        # Strides in elements
        stride_out_row = output.stride(0)
        stride_out_col = output.stride(1)
        stride_in_row = expert_outputs.stride(0)
        stride_in_col = expert_outputs.stride(1)

        # Launch Triton kernel with a 2D grid over rows and columns
        grid = (M, H)
        scatter_add_dim0_elementwise_kernel[grid](
            output, token_indices, expert_outputs,
            M, N, H,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            num_warps=4,  # reasonable default for simple elementwise work
        )
        return output


def run(*args):
    return ModelNew()(*args)
