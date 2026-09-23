import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per index (i). Vectorize across hidden_size in chunks of BLOCK_SIZE=256.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension
):
    pid = tl.program_id(axis=0)  # which expert output row to process
    if pid >= num_selected:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + pid)

    # Defensive bound check (indices are generated in [0, rows))
    if idx < 0 or idx >= rows:
        return

    # Loop over hidden dimension in chunks
    base_out = idx * hidden
    for col_start in range(0, hidden, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load the chunk from expert_outputs row pid
        exp_off = pid * hidden + col_start
        vals = tl.load(expert_ptr + exp_off + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

        # Atomic add into output
        tl.atomic_add(output_ptr + base_out + offs, vals, mask=mask)


def _choose_launch_params(hidden: int):
    # Heuristic tuning for Triton launch
    if hidden < 256:
        return 2, 2  # num_warps, num_stages
    elif hidden < 512:
        return 4, 2
    else:
        return 8, 2


def triton_scatter_add(final_hidden_states: torch.Tensor,
                       expert_outputs: torch.Tensor,
                       token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton-based scatter-add equivalent to torch.index_add(dim=0, index=token_indices, source=expert_outputs)
    into a cloned final_hidden_states. Returns the updated output.
    """
    assert TRITON_AVAILABLE, "Triton is not available."
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be CUDA tensors."
    assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Use bfloat16 tensors."

    rows = final_hidden_states.shape[0]
    hidden = final_hidden_states.shape[1]
    num_selected = expert_outputs.shape[0]

    # Clone to avoid modifying input in-place
    output = final_hidden_states.clone()

    # Triton requires int32 indices for pointer arithmetic
    indices = token_indices.to(torch.int32)

    # Choose launch params based on hidden size
    num_warps, num_stages = _choose_launch_params(hidden)

    # Launch grid: one program per selected index
    grid = (num_selected,)

    scatter_add_rows_kernel[grid](
        output, expert_outputs, indices,
        rows, hidden, num_selected,
        BLOCK_SIZE=256,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Run Triton scatter-add; keep host operations minimal and Triton-only for computation
        if TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            return triton_scatter_add(final_hidden_states, expert_outputs, token_indices)
        else:
            # Fallback: PyTorch implementation if Triton/CUDA not available
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output


def run(*args):
    return ModelNew()(*args)
