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
    output_ptr,         # *ptr to output [rows, hidden], dtype=bfloat16
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden], dtype=bfloat16
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size for hidden dimension (e.g., 256)
):
    pid = tl.program_id(axis=0)  # one program per selected index
    if pid >= num_selected:
        return

    # Load target row index for this selected token (int32)
    row_idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks
    for off in range(0, hidden, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load the expert row chunk (bfloat16)
        expert_row_ptr = expert_ptr + pid * hidden + offs
        vals = tl.load(expert_row_ptr, mask=mask, other=0.0)  # bfloat16

        # Compute destination pointer for the corresponding output row
        dest_ptr = output_ptr + row_idx * hidden + offs

        # Atomic add the chunk
        tl.atomic_add(dest_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Perform atomic accumulation of expert outputs back to token positions.
        - final_hidden_states: [batch_size * seq_len, hidden_size], bfloat16
        - expert_outputs: [num_selected_tokens, hidden_size], bfloat16
        - token_indices: [num_selected_tokens], int64 (will be cast to int32)
        Returns:
        - output: [batch_size * seq_len, hidden_size], bfloat16 (accumulated)
        """
        # Ensure inputs are on the same CUDA device and dtype
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtypes must be bfloat16."

        # Clone to avoid modifying input in-place
        output = final_hidden_states.clone()

        # Cast token_indices to int32 for Triton (kernel expects int32 indices)
        indices_i32 = token_indices.to(torch.int32)

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        # Use BLOCK_SIZE=256 and more warps to improve throughput without overcommitting
        scatter_add_rows_kernel[grid](
            output,          # output_ptr
            expert_outputs,  # expert_ptr
            indices_i32,     # indices_ptr (int32)
            rows,            # rows
            hidden,          # hidden
            num_selected,    # num_selected_tokens
            BLOCK_SIZE=256,
            num_warps=4,     # slightly higher warp count to improve utilization
            num_stages=1,
        )

        return output


def run(*args):
    return ModelNew()(*args)
