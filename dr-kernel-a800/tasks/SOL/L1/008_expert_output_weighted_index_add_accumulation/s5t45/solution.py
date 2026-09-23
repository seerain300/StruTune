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
    BLOCK_SIZE: tl.constexpr,  # vectorization chunk along hidden dim
):
    pid = tl.program_id(axis=0)
    if pid >= num_selected:
        return

    # Load target row index for this selected token
    row = tl.load(indices_ptr + pid)
    if row < 0 or row >= rows:
        return

    # Iterate over hidden dimension in BLOCK_SIZE chunks
    for off in range(0, hidden, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden

        # Load the expert output chunk
        expert_offs = pid * hidden + cols
        expert_vals = tl.load(expert_ptr + expert_offs, mask=mask, other=0.0)

        # Compute output pointer for the target row and corresponding columns
        out_offs = row * hidden + cols
        # Atomic add the chunk into the output row
        tl.atomic_add(output_ptr + out_offs, expert_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Clone to avoid modifying input in-place for reference correctness
        output = final_hidden_states.clone()

        # Triton kernel expects bf16; ensure dtype compatibility
        if output.dtype != torch.bfloat16:
            output = output.to(torch.bfloat16)
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)

        # Triton requires int32 for index computations in kernels
        indices_i32 = token_indices.to(torch.int32)

        # Shapes
        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Heuristic: use BLOCK_SIZE=256 for good vectorization; loops handle tails.
        BLOCK_SIZE = 256

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            indices_i32,
            rows,
            hidden,
            num_selected,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,  # good default for this type of per-row kernel
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
