import torch
import triton
import triton.language as tl


@triton.jit
def scatter_add_dim0_block_kernel(
    output_ptr,        # *bf16, shape [M, H], result tensor
    indices_ptr,       # *int32, shape [N], token_indices
    inputs_ptr,        # *bf16, shape [N, H], expert_outputs
    N,                 # int32, number of rows in inputs
    stride_out_row,    # int64, elements between rows of output
    stride_out_col,    # int64, elements between cols of output
    stride_in_row,     # int64, elements between rows of inputs
    stride_in_col,     # int64, elements between cols of inputs
    H: tl.constexpr,   # hidden_size (vector length), compile-time for tl.arange
    BLOCK_ROWS: tl.constexpr,  # number of rows handled per program
):
    # Each program handles a block of rows
    pid = tl.program_id(axis=0)
    # Compute the row range this program will process
    start_row = pid * BLOCK_ROWS
    row_ids = start_row + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS] vector of row indices
    mask_rows = row_ids < N

    # Load token_indices[i] for each row in this block
    idxs = tl.load(indices_ptr + row_ids, mask=mask_rows, other=0)  # int32

    # For each row in the block, load the corresponding expert vector and atomic add to output
    # We use a loop over rows to ensure each row is processed independently and safely.
    for k in range(BLOCK_ROWS):
        # Skip if this row is out of bounds
        if not (start_row + k < N):
            continue
        # Compute column pointers for this row: output[ idxs[k], :] and inputs[ k, :]
        out_ptrs = output_ptr + idxs[k] * stride_out_row + tl.arange(0, H) * stride_out_col
        in_ptrs = inputs_ptr + k * stride_in_row + tl.arange(0, H) * stride_in_col
        vals = tl.load(in_ptrs)  # bfloat16 vector of length H
        tl.atomic_add(out_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton path: ensure tensors are on CUDA
        if (not final_hidden_states.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
            # Fallback to PyTorch for safety (though we should use Triton on CUDA)
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Preserve original semantics: clone the accumulator (random init)
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

        # Launch Triton kernel: one program per block of rows
        BLOCK_ROWS = 256  # tuneable; 256 works well on many GPUs
        grid = (triton.cdiv(N, BLOCK_ROWS),)

        scatter_add_dim0_block_kernel[grid](
            output, token_indices, expert_outputs,
            N,
            stride_out_row, stride_out_col,
            stride_in_row, stride_in_col,
            H=H,
            BLOCK_ROWS=BLOCK_ROWS,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
