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
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension
):
    pid = tl.program_id(axis=0)  # which expert output row to process
    # bounds check (in case grid > num_selected)
    if pid >= num_selected:
        return

    # load target row index
    idx = tl.load(indices_ptr + pid)  # int32

    # iterate over hidden dimension in chunks
    for offs in range(0, hidden, BLOCK_SIZE):
        col = offs + tl.arange(0, BLOCK_SIZE)
        mask = col < hidden

        # compute pointers for this chunk
        # output_ptr is [rows, hidden], row-major. For a fixed row idx, address is base + idx * hidden + col
        out_ptrs = output_ptr + idx * hidden + col
        # expert_ptr is [num_selected, hidden], row-major. For row pid, address is base + pid * hidden + col
        exp_ptrs = expert_ptr + pid * hidden + col

        # load chunk from expert_outputs
        exp_chunk = tl.load(exp_ptrs, mask=mask, other=0.0)

        # atomic-add chunk into output row
        tl.atomic_add(out_ptrs, exp_chunk, mask=mask)


# Host-side forward: ensure Triton-only computation
class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Fallback to PyTorch if Triton not available
        if not TRITON_AVAILABLE or final_hidden_states.device.type != 'cuda':
            # Clone to match original behavior, then torch.index_add
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Ensure dtypes and devices
        assert final_hidden_states.dtype == torch.bfloat16, "final_hidden_states must be bfloat16"
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bfloat16"
        assert token_indices.dtype == torch.long, "token_indices must be torch.long"
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda, "Inputs must be on CUDA device for Triton"

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Cast indices to int32 for Triton
        indices_i32 = token_indices.to(torch.int32)

        # Get shapes
        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Choose num_warps based on hidden size for better occupancy
        if hidden < 256:
            num_warps = 2
        elif hidden < 512:
            num_warps = 4
        else:
            num_warps = 8

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, indices_i32,
            rows, hidden, num_selected,
            BLOCK_SIZE=256,
            num_warps=num_warps,
            num_stages=2,
        )
        return output


def run(*args):
    return ModelNew()(*args)
