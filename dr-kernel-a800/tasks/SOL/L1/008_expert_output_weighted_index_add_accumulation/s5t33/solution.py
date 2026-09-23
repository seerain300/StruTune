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
    output_ptr,         # *ptr to output [rows, hidden], dtype: bf16
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden], dtype: bf16
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # int: number of rows in output (i.e., batch_size * seq_len)
    hidden,             # int: hidden_size (number of columns)
    num_selected,       # int: number of selected tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension (e.g., 256)
):
    pid = tl.program_id(axis=0)
    # Each program handles one selected token
    if pid >= num_selected:
        return

    # Load destination row index for this selected token
    idx = tl.load(indices_ptr + pid)
    # Safety: indices should be in [0, rows)
    if (idx < 0) or (idx >= rows):
        return

    # Base offsets for this row in output and for this selected token in expert
    base_out = idx * hidden
    base_exp = pid * hidden

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    for col_start in range(0, hidden, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load expert outputs for this chunk
        vals = tl.load(expert_ptr + base_exp + offs, mask=mask, other=0.0)

        # Atomic add into output
        tl.atomic_add(output_ptr + base_out + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Minimal host-side logic: clone final_hidden_states and launch Triton kernel
        output = final_hidden_states.clone()

        # Ensure token_indices are int32 for Triton
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        # Ensure tensors are CUDA and contiguous for Triton
        if TRITON_AVAILABLE and output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            rows = output.shape[0]
            hidden = output.shape[1]
            num_selected = expert_outputs.shape[0]

            # Grid: one program per selected token
            grid = (num_selected,)

            # Launch kernel with a stable configuration that performs well across shapes
            scatter_add_rows_kernel[grid](
                output,
                expert_outputs,
                token_indices,
                rows,
                hidden,
                num_selected,
                BLOCK_SIZE=256,
                num_warps=4,
                num_stages=2,
            )

        return output


def run(*args):
    return ModelNew()(*args)
