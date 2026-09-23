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
    BLOCK_SIZE: tl.constexpr,  # vectorization width along hidden dim (use 256)
):
    pid = tl.program_id(0)  # one program per selected index
    # Load destination row index for this program
    row_idx = tl.load(indices_ptr + pid).to(tl.int32)
    # Base offset for this row
    out_row_base = row_idx * hidden

    # Iterate over hidden dimension in chunks
    for h in range(0, hidden, BLOCK_SIZE):
        offs = h + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load current chunk of expert row (assume bfloat16)
        e_ptrs = expert_ptr + pid * hidden + offs
        e_vals = tl.load(e_ptrs, mask=mask, other=0.0)

        # Compute output pointers for this chunk
        o_ptrs = output_ptr + out_row_base + offs

        # Atomic add this chunk to the output row
        tl.atomic_add(o_ptrs, e_vals, mask=mask)


def triton_scatter_add(output: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Triton implementation of output.index_add_(dim=0, index=token_indices, source=expert_outputs) using atomic adds.
    Requires Triton and CUDA tensors; falls back to PyTorch if Triton/CUDA not available.
    """
    # Fallback to PyTorch if Triton/CUDA not available
    if (not TRITON_AVAILABLE) or (not output.is_cuda) or (not expert_outputs.is_cuda) or (not token_indices.is_cuda):
        return output.index_add_(0, token_indices, expert_outputs)

    rows = output.shape[0]
    hidden = output.shape[1]
    num_selected = expert_outputs.shape[0]

    # Triton prefers int32 indices
    indices_i32 = token_indices.to(torch.int32)

    # Launch kernel: one program per selected index
    grid = (num_selected,)
    scatter_add_rows_kernel[grid](
        output,                # output_ptr
        expert_outputs,        # expert_ptr
        indices_i32,           # indices_ptr
        rows,                  # rows
        hidden,                # hidden
        num_selected,          # num_selected_tokens
        BLOCK_SIZE=256,        # vectorization width along hidden
        num_warps=1,           # each program is light; 1 warp is sufficient
        num_stages=1,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match original semantics (reference clones before index_add)
        output = final_hidden_states.clone()
        # Triton scatter-add (fallback handled inside)
        output = triton_scatter_add(output, expert_outputs, token_indices)
        return output


def run(*args):
    return ModelNew()(*args)
