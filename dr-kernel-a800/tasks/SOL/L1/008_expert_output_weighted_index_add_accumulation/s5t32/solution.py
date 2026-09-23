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

    # Load target row index for this selected token
    row_idx = tl.load(indices_ptr + pid)
    if row_idx < 0 or row_idx >= rows:
        return

    # Iterate across hidden dimension in chunks of BLOCK_SIZE
    offset = 0
    while offset < hidden:
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden

        # Load source chunk: expert_outputs[pid, cols]
        src_ptrs = expert_ptr + pid * hidden + cols
        src_vals = tl.load(src_ptrs, mask=mask, other=0.0)  # bfloat16

        # Destination pointers for this row and columns
        dst_ptrs = output_ptr + row_idx * hidden + cols

        # Atomic add for this chunk
        tl.atomic_add(dst_ptrs, src_vals, mask=mask)

        offset += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(
        self,
        final_hidden_states: torch.Tensor,
        expert_outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Fallback if Triton unavailable or tensors not on CUDA
        if not TRITON_AVAILABLE or not final_hidden_states.is_cuda:
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Prepare output by cloning final_hidden_states
        output = final_hidden_states.clone()

        # Shapes
        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Triton expects int32 indices
        if token_indices.dtype != torch.int32:
            token_indices_32 = token_indices.to(torch.int32)
        else:
            token_indices_32 = token_indices

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)

        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            token_indices_32,
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
