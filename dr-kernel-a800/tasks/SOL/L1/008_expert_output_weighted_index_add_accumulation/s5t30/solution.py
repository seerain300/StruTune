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
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension (256)
):
    pid = tl.program_id(axis=0)
    # Safety guard in case of overlaunch (grid == num_selected, so typically not needed)
    if pid >= num_selected:
        return

    # Load token index for this program
    row_idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks
    for start in range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load expert chunk
        expert_vals = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)

        # Compute output offsets for this row
        out_offsets = row_idx * hidden + offs

        # Atomic add into output (bf16)
        tl.atomic_add(output_ptr + out_offsets, expert_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # If Triton/CUDA not available, fallback to PyTorch
        if not (TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Prepare shapes
        rows = final_hidden_states.shape[0]
        hidden = final_hidden_states.shape[1]
        num_selected = expert_outputs.shape[0]

        # Output buffer (clone to match reference semantics)
        output = final_hidden_states.clone()

        # Ensure dtype compatibility: output and expert_outputs are bfloat16
        if output.dtype != torch.bfloat16:
            output = output.to(torch.bfloat16)
        if expert_outputs.dtype != torch.bfloat16:
            expert_outputs = expert_outputs.to(torch.bfloat16)

        # Ensure token_indices are int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=4,
            num_stages=2
        )
        return output


def run(*args):
    return ModelNew()(*args)
