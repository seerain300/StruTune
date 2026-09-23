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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= num_selected:
        return

    # Load the target row index for this selected token
    row_index = tl.load(indices_ptr + pid)  # int32

    # Process the hidden dimension in chunks of BLOCK_SIZE
    for offs in range(0, hidden, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden

        # Compute pointers
        # Output: output[row_index, cols]
        out_ptrs = output_ptr + row_index * hidden + cols
        # Expert: expert_outputs[pid, cols]
        exp_ptrs = expert_ptr + pid * hidden + cols

        # Load expert outputs (bf16) and add atomically to output
        val = tl.load(exp_ptrs, mask=mask, other=0.0)
        tl.atomic_add(out_ptrs, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Clone to avoid modifying input
        output = final_hidden_states.clone()

        # Ensure dtypes/devices
        assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
        assert output.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Expect bfloat16 tensors."

        # Triton requires int32 indices for pointer arithmetic
        indices_i32 = token_indices.to(torch.int32)

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, indices_i32,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
