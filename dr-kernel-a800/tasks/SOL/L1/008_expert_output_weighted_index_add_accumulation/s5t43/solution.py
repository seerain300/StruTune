import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per index (i). Vectorize across hidden dimension in chunks of BLOCK_SIZE=256.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,         # *ptr to output [rows, hidden], dtype=bfloat16
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden], dtype=bfloat16
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens (number of atomics to perform)
    BLOCK_SIZE: tl.constexpr,  # chunk size for hidden dimension
):
    pid = tl.program_id(axis=0)  # which token index this program handles
    if pid >= num_selected:
        return

    # Load the destination row index for this selected token (int32)
    row_idx = tl.load(indices_ptr + pid)
    if row_idx < 0 or row_idx >= rows:
        return

    # Pointer base for the destination row in output
    out_row_base = output_ptr + row_idx * hidden

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    for h_start in range(0, hidden, BLOCK_SIZE):
        offs = h_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load the expert row chunk (experts are contiguous along hidden)
        expert_row_base = expert_ptr + pid * hidden
        vals = tl.load(expert_row_base + offs, mask=mask, other=0.0)

        # Atomic add into the output row at the corresponding positions
        tl.atomic_add(out_row_base + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure Triton is available and device is CUDA; otherwise fallback to PyTorch
        if not TRITON_AVAILABLE or final_hidden_states.device.type != 'cuda':
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Validate dtypes and contiguity
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype in (torch.int32, torch.int64), "token_indices must be int32 or int64"

        # Triton prefers int32 for indices; cast to int32
        indices = token_indices.to(torch.int32)

        # Clone output
        output = final_hidden_states.clone()

        # Shapes
        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, indices,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=1,  # one warp is sufficient given small per-program work
        )

        return output


def run(*args):
    return ModelNew()(*args)
