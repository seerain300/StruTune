import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per index (i). Vectorize across hidden_size in chunks of BLOCK_SIZE=256.
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
    pid = tl.program_id(axis=0)  # one program per selected token

    # Guard in case grid > num_selected (not strictly needed if grid == num_selected)
    if pid >= num_selected:
        return

    # Load target row index for this selected token
    index = tl.load(indices_ptr + pid)
    # Compute base offsets for this row and the selected token's expert chunk
    row_offset = index * hidden

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    h_start = 0
    while h_start < hidden:
        cols = h_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden

        # Load the chunk of the expert output for this token
        expert_chunk = tl.load(expert_ptr + pid * hidden + cols, mask=mask, other=0.0)

        # Atomic add into the output row
        tl.atomic_add(output_ptr + row_offset + cols, expert_chunk, mask=mask)

        h_start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-accelerated scatter-add:
        output[token_indices[i]] += expert_outputs[i] for all i
        """
        # Ensure device and dtype compatibility
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA"
        assert final_hidden_states.dtype == expert_outputs.dtype, "final_hidden_states and expert_outputs must have the same dtype"
        assert final_hidden_states.dtype in (torch.bfloat16, torch.float16, torch.float32), "Unsupported dtype"

        # Prepare output: clone final_hidden_states
        output = final_hidden_states.clone()

        # Cast token indices to int32 for Triton atomic_add
        indices_i32 = token_indices.to(torch.int32)

        # Number of tokens in the batch*seq dimension
        rows = final_hidden_states.shape[0]
        hidden = final_hidden_states.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel
        grid = (num_selected,)
        # Choose a robust configuration that worked well across varied shapes
        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            indices_i32,
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
