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
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int64/long)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension (256)
):
    pid = tl.program_id(axis=0)  # which expert output row to process
    if pid >= num_selected:
        return

    # load target row index (indices are long/int64)
    idx = tl.load(indices_ptr + pid)  # int64

    # iterate over hidden dimension in chunks
    for off in range(0, hidden, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)  # vector of column offsets
        mask = cols < hidden

        # compute flat pointers
        out_ptrs = output_ptr + idx * hidden + cols
        exp_ptrs = expert_ptr + pid * hidden + cols

        # load, atomic add, store
        exp_vals = tl.load(exp_ptrs, mask=mask, other=0.0)
        # atomic add into output
        tl.atomic_add(out_ptrs, exp_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != "cuda"):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices.to(torch.long), source=expert_outputs)
            return output

        # Prepare output as a clone of final_hidden_states
        output = final_hidden_states.clone()

        # Shapes: final_hidden_states [rows, hidden], expert_outputs [num_selected_tokens, hidden]
        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel: one program per index
        grid = (num_selected,)

        # Use fixed launch params that performed best across workloads
        scatter_add_rows_kernel[grid](
            output,                             # output_ptr
            expert_outputs,                    # expert_ptr
            token_indices,                     # indices_ptr (keep as long; Triton loads fine)
            rows,                              # rows
            hidden,                            # hidden
            num_selected,                      # num_selected
            BLOCK_SIZE=256,                    # chunk size
            num_warps=4,                       # moderate parallelism per program
            num_stages=2,                      # pipeline stages
        )
        return output


def run(*args):
    return ModelNew()(*args)
