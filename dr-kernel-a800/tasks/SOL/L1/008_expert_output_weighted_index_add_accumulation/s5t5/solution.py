import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs rows into output at token_indices rows
# We process the hidden_size dimension in vectorized chunks (BLOCK_SIZE) to improve throughput.
if TRITON_AVAILABLE:
    @triton.jit
    def _atomic_scatter_add_vec_kernel(
        out_ptr,         # *bf16, pointer to output tensor [n_rows, n_cols]
        idx_ptr,         # *i32, pointer to token_indices [n_indices]
        vec_ptr,         # *bf16, pointer to expert_outputs [n_indices, n_cols]
        n_indices: tl.constexpr,  # number of indices
        n_cols: tl.constexpr,     # hidden_size
        BLOCK_SIZE: tl.constexpr, # chunk size for columns
    ):
        pid = tl.program_id(axis=0)
        # Bounds check: if grid > n_indices (normally grid == n_indices), this is a no-op
        if pid >= n_indices:
            return

        # Load target row index
        idx = tl.load(idx_ptr + pid)  # int32

        # Process columns in chunks
        j = 0
        while j < n_cols:
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            # Compute linear offsets for vectorized load/store
            # out_row_ptr points to the start of the idx-th row
            out_row_ptr = out_ptr + idx * n_cols + cols
            # Load the vector chunk from expert_outputs row pid
            vec_chunk = tl.load(vec_ptr + pid * n_cols + cols, mask=mask, other=0.0)
            # Atomic add into output row
            tl.atomic_add(out_row_ptr, vec_chunk, mask=mask)
            j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate shapes
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D (n_rows, hidden_size)"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D (n_indices, hidden_size)"
        assert token_indices.dim() == 1, "token_indices must be 1D (n_indices,)"

        # If Triton/CUDA not available, fall back to PyTorch to preserve correctness
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            # Ensure dtypes and device alignment
            idx = token_indices.to(output.device)
            vec = expert_outputs.to(output.dtype)
            output.index_add_(0, idx, vec)
            return output

        # Ensure tensors are contiguous for simpler addressing
        out = final_hidden_states.clone()  # preserves original semantics
        # Cast token_indices to int32 for Triton
        idx_i32 = token_indices.to(torch.int32)

        # Shapes
        n_indices = expert_outputs.shape[0]
        n_rows, n_cols = out.shape

        # Choose a vectorized chunk size: up to 256, aligned to common sizes
        # This value can be tuned; 256 often works well. Use min of 256 and round up to next power of two for better vectorization,
        # but keep it simple and safe: 256 for typical hidden sizes in the provided workloads.
        BLOCK_SIZE = 256

        # Launch one program per index
        grid = (n_indices,)
        _atomic_scatter_add_vec_kernel[grid](
            out,                # out_ptr
            idx_i32,            # idx_ptr
            expert_outputs,     # vec_ptr
            n_indices=n_indices,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        return out


def run(*args):
    return ModelNew()(*args)
