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
    num_selected,       # number of selected tokens (equal to num_experts_per_tok * batch_seq_len)
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension
):
    pid = tl.program_id(axis=0)  # one program per selected token
    if pid >= num_selected:
        return

    # Load the target row index for this token (int32)
    row_index = tl.load(indices_ptr + pid)

    # Loop across hidden dimension in chunks of BLOCK_SIZE
    for col_start in range(0, hidden, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load the slice of expert_outputs for this token
        expert_vals = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)

        # Compute output pointers for this row slice
        output_row_ptr = output_ptr + row_index * hidden + offs

        # Atomic add the slice into the corresponding row
        tl.atomic_add(output_row_ptr, expert_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Minimal host logic: clone output, ensure dtypes/devices, and launch Triton.
        # Avoid any PyTorch tensor ops in host code beyond what's necessary.
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Triton requires CUDA tensors."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expected bfloat16 tensors."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Ensure token_indices are int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Use fixed meta-parameters known to be fast across varied shapes
        BLOCK_SIZE = 256
        num_warps = 4

        # Launch one program per selected token
        grid = (num_selected,)

        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            rows,
            hidden,
            num_selected,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=1,  # simple kernel; fewer stages are fine
        )

        return output


def run(*args):
    return ModelNew()(*args)
