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
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension (256)
):
    pid = tl.program_id(axis=0)
    # Each program handles one selected token i = pid
    # Load the destination row index for this token
    row_idx = tl.load(indices_ptr + pid)
    # Guard: if pid >= num_selected, early return (safety, though grid should match)
    if pid >= num_selected:
        return

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    for off in range(0, hidden, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden

        # Compute linear offsets for output[row_idx, cols] and expert[pid, cols]
        out_offsets = row_idx * hidden + cols
        ex_offsets = pid * hidden + cols

        # Load expert chunk and add to output with atomic
        ex_vals = tl.load(expert_ptr + ex_offsets, mask=mask, other=0.0)
        # Atomic add directly: accumulate expert into output
        tl.atomic_add(output_ptr + out_offsets, ex_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Output buffer as a clone of the input to match original semantics
        output = final_hidden_states.clone()

        # Cast token_indices to int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Compute shapes
        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output,                   # output_ptr
            expert_outputs,           # expert_ptr
            indices_i32,              # indices_ptr
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=4, num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
