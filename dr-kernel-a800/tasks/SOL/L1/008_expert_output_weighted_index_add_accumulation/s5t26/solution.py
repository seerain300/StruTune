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
    # Load target row index and expert row
    # Note: indices are int32
    idx = tl.load(indices_ptr + pid)  # row index to add to
    # Compute base pointers for this program
    # Row base in output
    row_base = idx * hidden
    # Expert base for this token
    expert_row_base = pid * hidden

    # Vectorized loop over hidden dimension in chunks of BLOCK_SIZE
    for start in range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load the current chunk of expert output (bf16)
        expert_vals = tl.load(expert_ptr + expert_row_base + offs, mask=mask, other=0.0)

        # Compute output addresses for this chunk and perform atomic add
        out_addrs = output_ptr + row_base + offs
        tl.atomic_add(out_addrs, expert_vals, mask=mask)


def triton_scatter_add(output: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Perform scatter-add: output[token_indices[i]] += expert_outputs[i]
    - output: (rows, hidden), bfloat16, device must be CUDA
    - expert_outputs: (num_selected_tokens, hidden), bfloat16, device must be CUDA
    - token_indices: (num_selected_tokens,), int32, device must be CUDA
    """
    assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "All tensors must be on CUDA for Triton."
    assert output.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "Dtype must be bfloat16."
    rows = output.shape[0]
    hidden = output.shape[1]
    num_selected = expert_outputs.shape[0]

    # Ensure contiguous
    output_c = output.contiguous()
    expert_c = expert_outputs.contiguous()
    indices_i32 = token_indices.to(torch.int32).contiguous()

    # Launch kernel: one program per selected token
    grid = (num_selected,)
    # Use a stable configuration that performs well across varied shapes
    scatter_add_rows_kernel[grid](
        output_c, expert_c, indices_i32,
        rows, hidden, num_selected,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )
    return output_c


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # If Triton is not available or tensors are not CUDA, fallback to torch.index_add
        if (not TRITON_AVAILABLE) or (final_hidden_states.device.type != 'cuda'):
            # Fallback path: ensure correctness without Triton
            output = final_hidden_states.clone()
            # index_add along dim=0
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Triton path
        output = triton_scatter_add(final_hidden_states, expert_outputs, token_indices)
        return output


def run(*args):
    return ModelNew()(*args)
