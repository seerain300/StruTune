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
    num_selected,       # number of expert outputs to add
    BLOCK_SIZE: tl.constexpr,  # hidden chunk size (e.g., 256)
):
    pid = tl.program_id(axis=0)
    in_bounds_i = pid < num_selected

    # Load target row index for this selected token
    index = tl.load(indices_ptr + pid, mask=in_bounds_i, other=0)
    index = tl.max(index, 0)  # defensive clamp

    start = 0
    while start < hidden:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = in_bounds_i & (offs < hidden)
        # Load a chunk of expert outputs for this selected token
        vals = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)
        # Atomic add into the corresponding row in output
        tl.atomic_add(output_ptr + index * hidden + offs, vals, mask=mask)
        start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to avoid modifying input in-place for correctness
        output = final_hidden_states.clone()

        # Ensure indices are int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # If Triton not available, fallback to torch implementation
        if not TRITON_AVAILABLE:
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Launch kernel: one program per selected token
        num_selected = expert_outputs.shape[0]
        grid = (num_selected,)
        BLOCK_SIZE = 256  # Vectorize across hidden dim; 256 works well for bfloat16 and common sizes
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            output.shape[0], output.shape[1], num_selected,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # reasonable default for simple vectorized loops
        )
        return output


def run(*args):
    return ModelNew()(*args)
