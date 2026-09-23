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

    # load target row index (int32)
    idx = tl.load(indices_ptr + pid)

    # loop over hidden dimension in chunks
    for col in range(0, hidden, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)                # [BLOCK_SIZE] column offsets
        mask = offs < hidden                                 # mask for tail

        # compute pointers
        out_row_ptr = output_ptr + idx * hidden
        out_col_ptr = out_row_ptr + offs

        expert_row_ptr = expert_ptr + pid * hidden
        expert_col_ptr = expert_row_ptr + offs

        # load chunk from expert_outputs and atomic_add to output
        vals = tl.load(expert_col_ptr, mask=mask, other=0.0)
        tl.atomic_add(out_col_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure dtypes/devices are correct
        if not (output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            raise RuntimeError("All tensors must be on CUDA for Triton execution.")
        if output.dtype != torch.bfloat16 or expert_outputs.dtype != torch.bfloat16:
            raise RuntimeError("Dtype must be bfloat16 for all tensors.")

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Cast indices to int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Choose num_warps based on hidden size for better occupancy
        if hidden < 256:
            num_warps = 2
        elif hidden < 512:
            num_warps = 4
        else:
            num_warps = 8

        # Launch Triton kernel: one program per index
        grid = (num_selected,)

        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            rows,
            hidden,
            num_selected,
            BLOCK_SIZE=256,
            num_warps=num_warps,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
