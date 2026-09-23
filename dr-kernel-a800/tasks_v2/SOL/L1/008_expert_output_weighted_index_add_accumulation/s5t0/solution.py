import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: perform scatter-add along dim=0 of a 2D tensor
# For each i in [0, n_indices):
#   idx = token_indices[i]
#   vec = expert_outputs[i, :]
#   output[idx, :] += vec
# Assumes output is initially zero or contains initial values (we'll clone final_hidden_states).
@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,         # *bf16, shape (n_rows, n_cols)
    idx_ptr,         # *i32, shape (n_indices,)
    vec_ptr,         # *bf16, shape (n_indices, n_cols)
    n_indices: tl.constexpr,  # number of indices (rows to add)
    n_cols: tl.constexpr,     # number of columns (hidden_size)
    BLOCK_SIZE: tl.constexpr, # chunk size for columns
):
    # Each program handles one index/row addition
    pid = tl.program_id(0)
    # Load the target row index for this program
    idx = tl.load(idx_ptr + pid)
    # Base pointers for this row
    # We assume out_ptr, vec_ptr are row-major and contiguous
    # Loop over columns in chunks of BLOCK_SIZE
    # Note: Triton supports vectorized atomic_add; if not, this still works with scalar loops
    for j in range(0, n_cols, BLOCK_SIZE):
        cols = j + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # Load current row chunk and expert chunk
        row_vec = tl.load(out_ptr + idx * n_cols + cols, mask=mask, other=0.0)
        vec_vec = tl.load(vec_ptr + pid * n_cols + cols, mask=mask, other=0.0)
        # Add and atomic add back to output
        new_vec = row_vec + vec_vec
        tl.atomic_add(out_ptr + idx * n_cols + cols, new_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # If Triton not available or tensors are not on CUDA, fall back to PyTorch for correctness
        use_triton = TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda

        # Ensure dtypes and contiguity
        # We will operate in-place on an output tensor initialized from final_hidden_states
        # Clone to be safe and to match the original behavior (modifying output)
        output = final_hidden_states.clone()

        if not use_triton:
            # Fallback: PyTorch index_add along dim=0
            # Note: index_add returns a new tensor; since we cloned, we can just add into output
            # However, index_add mutates along the dim, so we can do:
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Triton path
        # Prepare pointers
        # Triton expects int32 indices; cast safely (token_indices are in range [0, batch_seq_len], which fits int32)
        idx_i32 = token_indices.to(torch.int32)

        # Ensure contiguous memory
        # (Triton works fine with PyTorch contiguous tensors; we can pass as-is)
        # But explicit checks: make sure both are contiguous
        # (get_inputs produces contiguous tensors; keep it robust)
        # If not contiguous, make them contiguous
        if not final_hidden_states.is_contiguous():
            final_hidden_states = final_hidden_states.contiguous()
        if not expert_outputs.is_contiguous():
            expert_outputs = expert_outputs.contiguous()
        if not idx_i32.is_contiguous():
            idx_i32 = idx_i32.contiguous()

        n_rows = output.shape[0]  # batch_seq_len
        n_cols = output.shape[1]  # hidden_size
        n_indices = expert_outputs.shape[0]

        # Launch Triton kernel: 1D grid over n_indices
        # Choose BLOCK_SIZE as 128 or 256; 128 is fine and commonly used
        BLOCK_SIZE = 128
        grid = (n_indices,)

        _scatter_add_rows_kernel[grid](
            output, idx_i32, expert_outputs, n_indices, n_cols, BLOCK_SIZE,
            num_warps=4,  # reasonable default for simple elementwise kernel
        )

        return output


def run(*args):
    return ModelNew()(*args)
