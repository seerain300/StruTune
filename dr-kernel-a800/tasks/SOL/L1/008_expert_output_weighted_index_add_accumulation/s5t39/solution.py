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
    num_selected,       # number of selected tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size along hidden dimension
):
    # program id corresponds to the selected token index
    pid = tl.program_id(0)
    # guard in case grid > num_selected (though we set grid=num_selected)
    if pid >= num_selected:
        return

    # compute the row index where this contribution should be added
    row = tl.load(indices_ptr + pid)
    if row < 0 or row >= rows:
        return  # index out of bounds; do nothing

    # base offsets for this row
    base_out = row * hidden
    base_exp = pid * hidden

    # iterate over hidden dimension in chunks of BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    for h in range(0, hidden, BLOCK_SIZE):
        mask = (h + offs) < hidden
        # load a chunk of expert outputs
        vals = tl.load(expert_ptr + base_exp + h + offs, mask=mask, other=0.0)
        # atomic add to corresponding positions in output
        tl.atomic_add(output_ptr + base_out + h + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to match reference semantics
        output = final_hidden_states.clone()

        # Ensure dtypes/devices are consistent
        # The original tensors are bfloat16; we keep that for output and expert_outputs.
        # token_indices should be int32 for Triton indexing.
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per selected token
        num_selected = expert_outputs.shape[0]
        rows = output.shape[0]
        hidden = output.shape[1]

        # Choose a reasonable BLOCK_SIZE. 256 works well for many hidden sizes.
        BLOCK_SIZE = 256

        # Grid size equals number of selected tokens
        grid = (num_selected,)

        scatter_add_rows_kernel[grid](
            output,                  # output_ptr
            expert_outputs,          # expert_ptr
            token_indices,           # indices_ptr
            rows, hidden, num_selected,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output


def run(*args):
    return ModelNew()(*args)
