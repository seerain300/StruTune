import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add in chunks to improve vectorization
if TRITON_AVAILABLE:
    @triton.jit
    def scatter_add_experts_kernel(
        out_ptr,          # *bf16, shape (n_rows, n_cols)
        idx_ptr,          # *i32, shape (n_indices,)
        vec_ptr,          # *bf16, shape (n_indices, n_cols)
        n_indices: tl.int32,
        n_cols: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        if pid >= n_indices:
            return

        # Target row index for this element
        idx = tl.load(idx_ptr + pid)  # int32

        # Iterate over hidden_size in chunks of BLOCK_SIZE
        j = 0
        while j < n_cols:
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols

            # Load the chunk from expert_outputs[pid, :]
            vec_chunk = tl.load(vec_ptr + pid * n_cols + cols, mask=mask, other=0.0)

            # Atomic-add into the corresponding output row
            out_row_ptr = out_ptr + idx * n_cols + cols
            tl.atomic_add(out_row_ptr, vec_chunk, mask=mask)

            j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Fallback to PyTorch if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices.to(output.device, dtype=torch.long), expert_outputs.to(output.dtype))
            return output

        # Validate shapes/dtypes
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D (n_rows, hidden_size)"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D (n_indices, hidden_size)"
        assert token_indices.dim() == 1, "token_indices must be 1D (n_indices,)"

        n_rows, n_cols = final_hidden_states.shape
        n_indices = token_indices.shape[0]

        # Prepare output: clone to match original behavior
        output = final_hidden_states.clone()

        # Cast token_indices to int32 for Triton
        idx32 = token_indices.to(torch.int32)

        # Choose BLOCK_SIZE for vectorization; 256 works well for typical hidden sizes
        BLOCK_SIZE = 256

        # Launch Triton kernel: one program per index
        grid = (n_indices,)

        scatter_add_experts_kernel[grid](
            output,
            idx32,
            expert_outputs,
            n_indices,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
