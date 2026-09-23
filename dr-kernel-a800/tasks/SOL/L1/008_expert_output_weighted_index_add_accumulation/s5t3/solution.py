import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_bf16_kernel(
    out_ptr,            # *bf16, shape (n_rows, n_cols)
    idx_ptr,            # *i32, shape (n_indices,)
    vec_ptr,            # *bf16, shape (n_indices, n_cols)
    n_indices: tl.int32,
    n_cols: tl.int32,
    stride_out_row: tl.int32,  # typically n_cols
    stride_out_col: tl.int32,  # typically 1
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per index
    if pid >= n_indices:
        return

    # Load target row index for this program
    row_idx = tl.load(idx_ptr + pid)  # int32

    # Iterate over columns in chunks of BLOCK_SIZE and atomic add
    j = 0
    while j < n_cols:
        cols = j + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # Compute base pointers for output row and vector chunk
        out_row_ptrs = out_ptr + row_idx * stride_out_row + cols * stride_out_col
        vec_ptrs = vec_ptr + pid * n_cols + cols

        # Load vector chunk (bf16), mask out-of-range with zeros
        vec_chunk = tl.load(vec_ptrs, mask=mask, other=0.0)

        # Atomic add into the output row at positions cols
        tl.atomic_add(out_row_ptrs, vec_chunk, mask=mask)

        j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate inputs
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D (n_rows, hidden_size)"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D (n_indices, hidden_size)"
        assert token_indices.dim() == 1, "token_indices must be 1D (n_indices,)"

        # Fallback to PyTorch if not CUDA
        if final_hidden_states.device.type != "cuda":
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices.to(output.device), expert_outputs.to(output.dtype))
            return output

        # Ensure dtypes and device are consistent; keep bf16
        out = final_hidden_states.clone().contiguous()
        expert_outputs = expert_outputs.contiguous()
        idx_i32 = token_indices.to(torch.int32)

        n_indices = expert_outputs.shape[0]
        n_cols = expert_outputs.shape[1]

        # Choose a block size for vectorized column processing.
        # Using 256 provides good throughput; loop handles any n_cols.
        BLOCK_SIZE = 256

        # Launch one program per index
        grid = (n_indices,)

        # Use 8 warps for better parallelism on chunked vector operations
        _scatter_add_rows_bf16_kernel[grid](
            out, idx_i32, expert_outputs,
            n_indices, n_cols,
            out.stride(0), out.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        return out


def run(*args):
    return ModelNew()(*args)
