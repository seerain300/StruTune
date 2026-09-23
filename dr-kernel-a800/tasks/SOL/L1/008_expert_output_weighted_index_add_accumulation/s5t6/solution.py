import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _scatter_add_rows_kernel(
    out_ptr,           # *bf16, pointer to output tensor (n_rows, n_cols)
    idx_ptr,           # *i32, pointer to token_indices (n_indices,)
    vec_ptr,           # *bf16, pointer to expert_outputs (n_indices, n_cols)
    n_indices: tl.int32,
    n_cols: tl.int32,  # hidden_size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= n_indices:
        return

    # Load target row index for this expert output
    idx = tl.load(idx_ptr + pid)  # int32

    # Iterate over columns in chunks and atomic add
    j = 0
    while j < n_cols:
        cols = j + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # Load a chunk of the vector for this index
        vec_chunk = tl.load(vec_ptr + pid * n_cols + cols, mask=mask, other=0.0)
        # Compute destination addresses and atomic add
        out_row_ptrs = out_ptr + idx * n_cols + cols
        tl.atomic_add(out_row_ptrs, vec_chunk, mask=mask)
        j += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Validate shapes
        assert final_hidden_states.dim() == 2, "final_hidden_states must be 2D (n_rows, hidden_size)"
        assert expert_outputs.dim() == 2, "expert_outputs must be 2D (n_indices, hidden_size)"
        assert token_indices.dim() == 1, "token_indices must be 1D (n_indices,)"

        # Fallback to PyTorch if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            token_indices_t = token_indices.to(output.device)
            # Ensure dtype matches
            if expert_outputs.dtype != output.dtype:
                expert_outputs = expert_outputs.to(dtype=output.dtype)
            # index_add supports long for indices; cast if needed
            if token_indices_t.dtype != torch.long:
                token_indices_t = token_indices_t.to(torch.long)
            output.index_add_(0, token_indices_t, expert_outputs.to(device=output.device))
            return output

        # Prepare output as a clone of the input buffer to match original behavior
        output = final_hidden_states.clone()

        # Ensure token_indices are int32 for Triton and on the correct device
        token_indices_i32 = token_indices.to(torch.int32).to(device=output.device)

        # Ensure expert_outputs are on the same device and dtype as output
        expert_outputs_cast = expert_outputs.to(dtype=output.dtype, device=output.device)

        # Launch Triton kernel: one program per index
        n_indices = token_indices_i32.numel()
        n_rows = final_hidden_states.shape[0]
        n_cols = final_hidden_states.shape[1]

        # Heuristic for BLOCK_SIZE based on hidden_size
        if n_cols >= 1024:
            BLOCK_SIZE = 256
        elif n_cols >= 256:
            BLOCK_SIZE = 128
        else:
            BLOCK_SIZE = 64

        _scatter_add_rows_kernel[(n_indices,)](
            output, token_indices_i32, expert_outputs_cast,
            n_indices, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
