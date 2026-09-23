import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per selected token index (i). Robust handling of any hidden_size via per-element masking.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # number of selected tokens
):
    # program id: one per selected token
    i = tl.program_id(axis=0)
    if i >= num_selected:
        return

    # load the destination row index for this selected token
    row_idx = tl.load(indices_ptr + i)
    if row_idx < 0 or row_idx >= rows:
        return

    # Iterate over hidden dimension with per-element masking
    # This ensures correctness for any hidden_size, including odd sizes.
    for j in range(0, hidden):
        # Compute pointers for output row[j] and expert[i, j]
        out_ptr = output_ptr + row_idx * hidden + j
        exp_ptr = expert_ptr + i * hidden + j

        # Load and atomic add for this element (masked always true, but keeps structure general)
        val = tl.load(exp_ptr)  # element is bfloat16
        tl.atomic_add(out_ptr, val)


def _run_triton(final_hidden_states: torch.Tensor,
                expert_outputs: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
    # Ensure dtypes and device compatibility
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors."
    assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16."
    assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16."
    assert token_indices.dtype == torch.int64, "token_indices must be torch.long (int64)."

    # Clone to avoid modifying input in-place
    output = final_hidden_states.clone()

    # Cast token indices to int32 for Triton (kernel expects int32)
    indices_i32 = token_indices.to(torch.int32)

    # Shapes
    rows = output.shape[0]
    hidden = output.shape[1]
    num_selected = expert_outputs.shape[0]

    # Launch kernel: one program per selected index
    grid = (num_selected,)

    # Use a modest number of warps; each program touches one row, contention is low.
    scatter_add_rows_kernel[grid](
        output, expert_outputs, indices_i32,
        rows, hidden, num_selected,
        num_warps=1,
        num_stages=1,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Triton forward: must use Triton kernels; no torch ops in host code.
        return _run_triton(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
