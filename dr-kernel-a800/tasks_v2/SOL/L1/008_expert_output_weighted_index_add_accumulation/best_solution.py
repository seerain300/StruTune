import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: For each i in [0, n_indices):
#   idx = token_indices[i]
#   vec = expert_outputs[i, :]
#   output[idx, :] += vec  (atomic add to handle duplicates)
@triton.jit
def _atomic_scatter_add_kernel(
    out_ptr,         # *bf16, shape (n_rows, n_cols), where n_rows = batch_size * seq_len
    idx_ptr,         # *i32, shape (n_indices,)
    vec_ptr,         # *bf16, shape (n_indices, n_cols)
    n_indices,       # int32
    n_cols,          # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= n_indices:
        return

    # Load target row index for this element
    idx = tl.load(idx_ptr + pid)  # int32

    # Iterate over columns in chunks of BLOCK_SIZE and atomic add
    j = 0
    while j < n_cols:
        cols = j + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # Load the vector chunk for this i (bf16)
        vec_chunk = tl.load(vec_ptr + pid * n_cols + cols, mask=mask, other=0.0)
        # Atomic add into the output row at row idx
        out_row_ptr = out_ptr + idx * n_cols + cols
        tl.atomic_add(out_row_ptr, vec_chunk, mask=mask)
        j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate shapes
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D (n_rows, hidden_size)"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D (n_indices, hidden_size)"
        assert token_indices.dim() == 1, "token_indices must be 1D (n_indices,)"

        # If Triton/CUDA not available, fallback to PyTorch for correctness
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            # Preserve original semantics: clone then index_add
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices.to(output.device), expert_outputs.to(output.dtype))
            return output

        # Clone to preserve initial values (same as original)
        output = final_hidden_states.clone()
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()

        # Prepare inputs for Triton
        # token_indices must be int32 for Triton
        idx_i32 = token_indices.to(torch.int32)

        # Dimensions
        n_indices = expert_outputs.shape[0]
        n_cols = output.shape[1]

        # Launch Triton kernel: one program per index
        BLOCK_SIZE = 128  # good default; adjust if hidden_size is very large
        grid = (n_indices,)

        _atomic_scatter_add_kernel[grid](
            output, idx_i32, expert_outputs, n_indices, n_cols, BLOCK_SIZE,
            num_warps=4,  # reasonable default for bandwidth-bound kernel
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
