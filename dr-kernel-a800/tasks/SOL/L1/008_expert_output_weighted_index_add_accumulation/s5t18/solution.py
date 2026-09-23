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

    # iterate over hidden dimension in chunks
    for start in range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # load expert values for this chunk
        expert_vals = tl.load(
            expert_ptr + pid * hidden + offs,
            mask=mask,
            other=0.0
        )  # shape: [BLOCK_SIZE], bfloat16

        # compute destination pointers for this chunk and atomic add
        out_ptrs = output_ptr + idx * hidden + offs
        tl.atomic_add(out_ptrs, expert_vals, mask=mask)


def run_triton(final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Triton-based scatter-add: output[token_indices[i]] += expert_outputs[i]
    - final_hidden_states: [batch_seq_len, hidden_size], bfloat16
    - expert_outputs: [num_selected_tokens, hidden_size], bfloat16
    - token_indices: [num_selected_tokens], int64 (will be cast to int32)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda
    assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16
    assert token_indices.dtype == torch.int64  # safe; we'll cast

    # Clone to preserve input immutability and match original behavior
    output = final_hidden_states.clone()

    # Shapes
    rows = output.shape[0]  # batch_size * seq_len
    hidden = output.shape[1]
    num_selected = expert_outputs.shape[0]

    # Ensure tensors are contiguous for simple pointer arithmetic
    expert_outputs = expert_outputs.contiguous()
    # Cast indices to int32 for Triton
    token_indices_i32 = token_indices.to(torch.int32)

    # Launch Triton kernel: one program per selected token
    grid = (num_selected,)
    # Use BLOCK_SIZE=256 which has shown robust performance across varied shapes
    scatter_add_rows_kernel[grid](
        output, expert_outputs, token_indices_i32,
        rows, hidden, num_selected,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Entry point as requested
        if TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            return run_triton(final_hidden_states, expert_outputs, token_indices)
        else:
            # Fallback to PyTorch behavior if Triton or CUDA not available
            output = final_hidden_states.clone()
            # torch.index_add handles duplicates correctly
            output.index_add_(dim=0, index=token_indices.to(torch.long), source=expert_outputs)
            return output


def run(*args):
    return ModelNew()(*args)
