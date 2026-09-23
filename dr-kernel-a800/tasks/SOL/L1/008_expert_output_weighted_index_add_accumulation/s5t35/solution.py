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
    BLOCK_SIZE: tl.constexpr,  # vectorization size over hidden, e.g., 256
):
    # Each program handles one index
    pid = tl.program_id(0)
    if pid >= num_selected:
        return

    # Load token index for this program
    row_idx = tl.load(indices_ptr + pid)  # int32

    # Vector of hidden offsets
    offs = tl.arange(0, BLOCK_SIZE)

    # Iterate over hidden dimension in chunks
    num_chunks = (hidden + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(0, num_chunks):
        start = chunk * BLOCK_SIZE
        h = start + offs  # vector of hidden offsets
        mask = h < hidden

        # Load the corresponding row from expert_outputs
        e = tl.load(expert_ptr + pid * hidden + h, mask=mask, other=0)

        # Compute output address and atomic add
        out_row_ptr = output_ptr + row_idx * hidden + h
        tl.atomic_add(out_row_ptr, e, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure minimal operations; clone to avoid modifying input
        output = final_hidden_states.clone()

        # Ensure dtypes and contiguity
        output = output.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton prefers int32 for indices on many kernels
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per selected token
        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        grid = (num_selected,)

        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
        )

        return output


def run(*args):
    return ModelNew()(*args)
