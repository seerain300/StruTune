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
    BLOCK_SIZE: tl.constexpr,  # vector width across hidden
):
    pid = tl.program_id(axis=0)
    if pid >= num_selected:
        return

    # Load target row index for this program
    idx = tl.load(indices_ptr + pid)

    # Loop over hidden dimension in chunks of BLOCK_SIZE
    for h in range(0, hidden, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        cols = h + col_offsets
        mask = cols < hidden

        # Compute pointers
        out_row_ptr = output_ptr + idx * hidden + cols
        exp_row_ptr = expert_ptr + pid * hidden + cols

        # Load values with mask; other=0 for out-of-range
        out_vals = tl.load(out_row_ptr, mask=mask, other=0.0)
        exp_vals = tl.load(exp_row_ptr, mask=mask, other=0.0)

        # Atomic add expert contribution to output
        tl.atomic_add(out_row_ptr, exp_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # If Triton not available or tensors not on CUDA, fall back to PyTorch implementation
        if not TRITON_AVAILABLE or not final_hidden_states.is_cuda:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Clone to match reference semantics
        output = final_hidden_states.clone()

        # Prepare inputs: ensure contiguity and dtypes
        if token_indices.dtype != torch.int32:
            token_indices_i32 = token_indices.to(torch.int32)
        else:
            token_indices_i32 = token_indices

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Grid: one program per selected token
        grid = (num_selected,)

        # Launch Triton kernel with a reasonable block size (256 works well for typical hidden sizes)
        scatter_add_rows_kernel[grid](
            output,               # output_ptr
            expert_outputs,       # expert_ptr
            token_indices_i32,    # indices_ptr
            rows,                 # rows
            hidden,               # hidden
            num_selected,         # num_selected
            BLOCK_SIZE=256,
        )

        return output


def run(*args):
    return ModelNew()(*args)
