import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Scatter-add kernel:
# For each index i in [0, n_indices):
#   idx = token_indices[i]
#   For j in [0, n_cols) step BLOCK_SIZE:
#       out[idx, j:j+BLOCK_SIZE] += expert_outputs[i, j:j+BLOCK_SIZE]
@triton.jit
def _scatter_add_vec_kernel(
    out_ptr,          # *bf16, pointer to output [n_rows, n_cols]
    idx_ptr,          # *i32, pointer to token indices [n_indices]
    vec_ptr,          # *bf16, pointer to expert outputs [n_indices, n_cols]
    n_indices,        # int32
    n_cols,           # int32
    BLOCK_SIZE: tl.constexpr,  # chunk size for columns (compile-time constant)
):
    pid = tl.program_id(axis=0)
    if pid >= n_indices:
        return

    # Load target row index (int32)
    idx = tl.load(idx_ptr + pid)

    # Process columns in chunks of BLOCK_SIZE
    # Triton will unroll this static loop if BLOCK_SIZE is constexpr
    for j in range(0, n_cols, BLOCK_SIZE):
        cols = j + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        # Compute flat pointers for this row slice
        out_row_ptr = out_ptr + idx * n_cols + cols
        vec_row_ptr = vec_ptr + pid * n_cols + cols

        # Load chunk of expert outputs (bf16), mask out-of-range columns
        vec_chunk = tl.load(vec_row_ptr, mask=mask, other=0.0)

        # Atomic add into output (handles duplicates correctly)
        tl.atomic_add(out_row_ptr, vec_chunk, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate shapes
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D (n_rows, hidden_size)"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D (n_indices, hidden_size)"
        assert token_indices.dim() == 1, "token_indices must be 1D (n_indices,)"

        # If Triton not available or tensors not on CUDA, fall back to PyTorch for correctness
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            # Ensure dtypes match for index_add
            if expert_outputs.dtype != output.dtype:
                expert_outputs = expert_outputs.to(output.dtype)
            if token_indices.device != output.device:
                token_indices = token_indices.to(output.device)
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Prepare output: clone to match original behavior
        out = final_hidden_states.clone()

        # Cast token indices to int32 for Triton
        idx_i32 = token_indices.to(torch.int32)

        # Shapes
        n_rows, n_cols = out.shape
        n_indices = expert_outputs.shape[0]

        # Choose BLOCK_SIZE: next power of two up to 256, but not exceeding n_cols too much
        # This improves vector utilization and reduces masked lanes.
        def _next_power_of_two(x: int) -> int:
            if x <= 1:
                return 1
            return 1 << ((x - 1).bit_length())

        # Cap at 256; if n_cols is smaller than 8, use at least 8 for vectorization
        BLOCK_SIZE = min(256, _next_power_of_two(n_cols))
        if BLOCK_SIZE < 8:
            BLOCK_SIZE = 8

        # Launch kernel: one program per index
        grid = (n_indices,)

        _scatter_add_vec_kernel[grid](
            out,                      # out_ptr
            idx_i32,                  # idx_ptr
            expert_outputs,           # vec_ptr
            n_indices,                # n_indices
            n_cols,                   # n_cols
            BLOCK_SIZE=BLOCK_SIZE,    # compile-time constant
            num_warps=4,              # balanced default for memory-bound ops
            num_stages=2,             # modest pipelining
        )

        return out


def run(*args):
    return ModelNew()(*args)
