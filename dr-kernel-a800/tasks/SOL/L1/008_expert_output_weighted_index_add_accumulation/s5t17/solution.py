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

    # load target row index
    idx = tl.load(indices_ptr + pid)  # int32

    # loop over hidden dimension in chunks
    # Each iteration processes a contiguous slice of the column dimension, enabling vectorized atomic adds.
    for offs in range(0, hidden, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden

        # load the chunk of expert output for this index
        # pointer to expert row: expert_ptr + pid * hidden + cols
        expert_offsets = pid * hidden + cols
        expert_vals = tl.load(expert_ptr + expert_offsets, mask=mask, other=0.0)  # bfloat16

        # compute output offsets: output_ptr + idx * hidden + cols
        out_offsets = idx * hidden + cols
        # atomic add into the output row
        tl.atomic_add(output_ptr + out_offsets, expert_vals, mask=mask)


def _run_triton_scatter_add(final_hidden_states: torch.Tensor,
                            expert_outputs: torch.Tensor,
                            token_indices: torch.Tensor) -> torch.Tensor:
    """
    Triton-based scatter-add that replaces torch.index_add along dim=0.
    Accumulates expert_outputs[i, :] into output[token_indices[i], :].
    """
    # Ensure device is CUDA and dtypes match; we keep bfloat16 as in the original.
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA for Triton."
    # Output: clone input to avoid modifying it in-place
    output = final_hidden_states.clone()

    rows = output.shape[0]
    hidden = output.shape[1]
    num_selected = expert_outputs.shape[0]

    # Ensure token_indices is int32 for Triton
    if token_indices.dtype != torch.int32:
        token_indices = token_indices.to(torch.int32)

    # Choose launch configuration based on hidden_size
    if hidden < 256:
        num_warps = 2
    elif hidden < 512:
        num_warps = 4
    else:
        num_warps = 8
    num_stages = 2

    # Launch one program per selected token
    grid = (num_selected,)

    # Run kernel
    scatter_add_rows_kernel[grid](
        output,                  # output_ptr
        expert_outputs,          # expert_ptr
        token_indices,           # indices_ptr
        rows,                    # rows
        hidden,                  # hidden
        num_selected,            # num_selected_tokens
        BLOCK_SIZE=256,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton implementation of the original run function.
        """
        # Use Triton kernel if available and tensors are on CUDA; otherwise fallback is implicitly not needed here
        # because the provided get_inputs returns CUDA tensors.
        return _run_triton_scatter_add(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
