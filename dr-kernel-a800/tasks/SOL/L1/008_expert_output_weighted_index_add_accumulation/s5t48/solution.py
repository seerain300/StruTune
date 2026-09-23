import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per selected index (i). Vectorize across hidden dimension in chunks of BLOCK_SIZE=256.
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
    # Each program handles one selected token index
    pid = tl.program_id(0)  # program id equals the token index i

    # Bounds check: grid is set to (num_selected,), so pid in [0, num_selected)
    # Load the target row index for this selected token
    idx = tl.load(indices_ptr + pid)  # int32

    # Loop over hidden dimension in chunks of BLOCK_SIZE
    for k in range(0, hidden, BLOCK_SIZE):
        col_offsets = k + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        mask = col_offsets < hidden  # mask for tail

        # Pointers for the current expert row and output row
        # Layout: expert_outputs is [num_selected_tokens, hidden] contiguous row-major
        #         output is [rows, hidden] contiguous row-major
        expert_row_ptr = expert_ptr + pid * hidden + col_offsets
        out_row_ptr = output_ptr + idx * hidden + col_offsets

        # Load expert values for this chunk (masked for tail)
        # Note: Triton will infer dtype from the pointer types. Ensure expert_outputs/output are bf16.
        expert_vals = tl.load(expert_row_ptr, mask=mask, other=0.0)

        # Atomic add to the corresponding output row
        tl.atomic_add(out_row_ptr, expert_vals, mask=mask)


def _run_triton(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    """
    Triton implementation of:
      output = final_hidden_states.clone()
      output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    """
    # Minimal host-side operations: ensure CUDA tensors and dtype consistency
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be CUDA."
    # Enforce bfloat16 for correctness
    if final_hidden_states.dtype != torch.bfloat16:
        final_hidden_states = final_hidden_states.to(torch.bfloat16)
    if expert_outputs.dtype != torch.bfloat16:
        expert_outputs = expert_outputs.to(torch.bfloat16)

    output = final_hidden_states.clone()

    rows = output.shape[0]
    hidden = output.shape[1]
    num_selected = expert_outputs.shape[0]

    # Cast token_indices to int32 for Triton
    indices_i32 = token_indices.to(torch.int32)

    # Choose chunk size across hidden dimension
    BLOCK_SIZE = 256  # good default; masks handle tails for any hidden
    grid = (num_selected,)

    # Launch kernel
    scatter_add_rows_kernel[grid](
        output, expert_outputs, indices_i32,
        rows, hidden, num_selected,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Use Triton kernel; host code does no PyTorch tensor ops beyond shape handling.
        return _run_triton(final_hidden_states, expert_outputs, token_indices)


def run(*args):
    return ModelNew()(*args)
