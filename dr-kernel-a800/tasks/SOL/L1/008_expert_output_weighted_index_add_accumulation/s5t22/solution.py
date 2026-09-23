import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per selected token (index). Vectorize across hidden_size in chunks of BLOCK_SIZE=256.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,         # *ptr to output [rows, hidden] (bfloat16)
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden] (bfloat16)
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension (256)
):
    pid = tl.program_id(0)  # each program handles one selected token
    # Load target row index for this selected token
    row_idx = tl.load(indices_ptr + pid)  # int32

    # Loop over hidden dimension in chunks of BLOCK_SIZE and atomic-add each chunk
    h_start = 0
    while h_start < hidden:
        offs = h_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load expert values for this token at current chunk (bf16)
        expert_vals = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)

        # Compute destination pointers for the target row and atomic-add
        out_ptrs = output_ptr + row_idx * hidden + offs
        tl.atomic_add(out_ptrs, expert_vals, mask=mask)

        h_start += BLOCK_SIZE


def triton_scatter_add(final_hidden_states: torch.Tensor,
                       expert_outputs: torch.Tensor,
                       token_indices: torch.Tensor) -> torch.Tensor:
    """
    Launches the Triton kernel to perform scatter-add:
      output[token_indices[i]] += expert_outputs[i] for all i.
    Assumes:
      - final_hidden_states is a clone of the original buffer (to be updated in-place).
      - expert_outputs shape: [num_selected_tokens, hidden_size], dtype: bfloat16
      - token_indices shape: [num_selected_tokens], dtype: int32
      - Tensors are on CUDA device and contiguous.
    """
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
    assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16 for this kernel."
    assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16 for this kernel."
    assert token_indices.dtype == torch.int32, "token_indices must be int32 for this kernel."

    rows = final_hidden_states.shape[0]
    hidden = final_hidden_states.shape[1]
    num_selected = expert_outputs.shape[0]

    # Ensure contiguous for pointer arithmetic
    output = final_hidden_states  # we'll mutate in-place via kernel atomic adds
    # Launch kernel: one program per selected token
    grid = (num_selected,)
    scatter_add_rows_kernel[grid](
        output,
        expert_outputs,
        token_indices,
        rows,
        hidden,
        num_selected,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Triton path (CUDA). Fallback to PyTorch if Triton unavailable or tensors on CPU.
        if TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            # Triton kernel mutates final_hidden_states in-place
            return triton_scatter_add(final_hidden_states, expert_outputs, token_indices)
        else:
            # Fallback: pure PyTorch scatter-add to ensure correctness on CPU or without Triton.
            output = final_hidden_states.clone()
            # torch.index_add along dim=0: add each row slice
            output.index_add_(dim=0, index=token_indices.to(torch.long), source=expert_outputs)
            return output


def run(*args):
    return ModelNew()(*args)
