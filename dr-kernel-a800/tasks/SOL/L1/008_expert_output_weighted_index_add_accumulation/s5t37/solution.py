import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per selected token i. Vectorize across hidden dimension in chunks of BLOCK_SIZE=256.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # vector width over hidden dim
):
    pid = tl.program_id(0)  # which selected token we process
    if pid >= num_selected:
        return

    # destination row index for this selected token (int32)
    dest_row = tl.load(indices_ptr + pid)

    # Loop over hidden dimension in chunks of BLOCK_SIZE
    for col_start in range(0, hidden, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load expert chunk for this token
        ep = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)

        # Compute output pointers for the destination row and atomic add
        out_ptrs = output_ptr + dest_row * hidden + offs
        tl.atomic_add(out_ptrs, ep, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-accelerated scatter-add:
          output[token_indices[i]] += expert_outputs[i]
        Returns:
          output tensor of shape [batch_size * seq_len, hidden_size]
        """
        # Ensure inputs are on GPU and Triton is available
        assert TRITON_AVAILABLE, "Triton is not available"
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be on CUDA"

        rows = final_hidden_states.shape[0]
        hidden = final_hidden_states.shape[1]
        num_selected = expert_outputs.shape[0]

        # Initialize output to zeros (avoid clone; we'll accumulate with atomic adds)
        output = torch.zeros_like(final_hidden_states, dtype=expert_outputs.dtype, device=final_hidden_states.device)

        # Prepare indices as int32
        indices = token_indices.to(torch.int32)

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, indices,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=8,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
