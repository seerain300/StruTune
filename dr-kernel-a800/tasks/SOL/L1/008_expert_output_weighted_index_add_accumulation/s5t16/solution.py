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
    # guard against out-of-range indices (indices should be [0, rows))
    if idx < 0 or idx >= rows:
        return

    # base offsets for the current index
    out_row_base = idx * hidden
    exp_row_base = pid * hidden

    # iterate over hidden dimension in chunks
    for off in range(0, hidden, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden
        out_ptrs = output_ptr + out_row_base + offs
        src_ptrs = expert_ptr + exp_row_base + offs

        # load source chunk and atomic add into destination
        vals = tl.load(src_ptrs, mask=mask, other=0.0)
        tl.atomic_add(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Clone to avoid modifying the input buffer
        output = final_hidden_states.clone()
        # Triton requires CUDA and int32 indices for performance and simplicity
        assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
        token_indices_i32 = token_indices.to(torch.int32)

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Choose launch parameters based on hidden_size
        if hidden < 256:
            num_warps = 2
            num_stages = 2
        elif hidden < 512:
            num_warps = 4
            num_stages = 2
        else:
            num_warps = 8
            num_stages = 2

        # Launch one program per index
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices_i32,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
