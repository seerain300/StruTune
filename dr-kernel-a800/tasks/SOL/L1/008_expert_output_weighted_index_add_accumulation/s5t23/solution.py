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
    output_ptr,         # *ptr to output [rows, hidden] (bfloat16)
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden] (bfloat16)
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension (256)
):
    pid = tl.program_id(0)  # one program per selected token
    # Safety guard (shouldn't be needed if grid matches num_selected)
    if pid >= num_selected:
        return

    # Load the target row index for this token (int32)
    row_idx = tl.load(indices_ptr + pid)
    if row_idx < 0 or row_idx >= rows:
        return

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    for h_start in range(0, hidden, BLOCK_SIZE):
        h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
        mask = h_offsets < hidden

        # Compute linear offsets (row-major layout)
        out_offsets = row_idx * hidden + h_offsets
        exp_off = pid * hidden + h_offsets

        # Load expert chunk and atomically add to output
        chunk_exp = tl.load(expert_ptr + exp_off, mask=mask, other=0.0)
        tl.atomic_add(output_ptr + out_offsets, chunk_exp, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
        - final_hidden_states: [batch_seq_len, hidden_size] bfloat16 (input buffer to be updated)
        - expert_outputs: [num_selected_tokens, hidden_size] bfloat16
        - token_indices: [num_selected_tokens] int64 (PyTorch default)
        Returns updated final_hidden_states with expert contributions added.
        """
        # Fallback to PyTorch if Triton unavailable or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != 'cuda'):
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices.to(torch.long), source=expert_outputs)
            return output

        # Ensure contiguity and dtype consistency
        output = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        # Triton expects int32 indices
        token_indices = token_indices.to(torch.int32).contiguous()

        rows = final_hidden_states.shape[0]
        hidden = final_hidden_states.shape[1]
        num_selected = token_indices.shape[0]

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, token_indices,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=4, num_stages=2
        )
        return output


def run(*args):
    return ModelNew()(*args)
