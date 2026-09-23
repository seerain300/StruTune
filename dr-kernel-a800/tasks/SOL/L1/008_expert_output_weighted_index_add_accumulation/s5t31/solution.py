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
    if pid >= num_selected:
        return

    # Load the target row index for this selected token
    row_index = tl.load(indices_ptr + pid)  # int32
    if row_index < 0 or row_index >= rows:
        return  # defensive check, inputs are expected valid

    # Iterate over hidden dimension in chunks
    for offs in range(0, hidden, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden

        # Load expert values for this chunk (bfloat16)
        expert_vals = tl.load(expert_ptr + pid * hidden + cols, mask=mask, other=0.0)

        # Atomic add into output[row_index, cols]
        out_row_ptr = output_ptr + row_index * hidden + cols
        tl.atomic_add(out_row_ptr, expert_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Triton path: ensure Triton and CUDA available
        if TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            # Clone to match original semantics
            output = final_hidden_states.clone()

            # Kernel expects int32 indices
            if token_indices.dtype != torch.int32:
                token_indices_i32 = token_indices.to(torch.int32)
            else:
                token_indices_i32 = token_indices

            rows = output.shape[0]
            hidden = output.shape[1]
            num_selected = expert_outputs.shape[0]

            # Launch Triton kernel: one program per selected token
            grid = (num_selected,)
            scatter_add_rows_kernel[grid](
                output,                # output_ptr
                expert_outputs,        # expert_ptr
                token_indices_i32,     # indices_ptr
                rows,                  # rows
                hidden,                # hidden
                num_selected,          # num_selected
                BLOCK_SIZE=256,        # vectorization width across hidden dimension
                num_warps=4,           # balanced for many shapes
                num_stages=2,          # typical choice
            )
            return output
        else:
            # Fallback to PyTorch index_add when Triton/CUDA not available
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices.to(torch.long), expert_outputs)
            return output


def run(*args):
    return ModelNew()(*args)
