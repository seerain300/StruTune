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

    # Load token index for this program
    idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks
    for start in range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Compute pointers for output and expert rows
        out_ptrs = output_ptr + idx * hidden + offs
        exp_ptrs = expert_ptr + pid * hidden + offs

        # Load values (bf16) with masking for tail
        vals = tl.load(exp_ptrs, mask=mask, other=0.0)

        # Atomic add into output (bf16)
        tl.atomic_add(out_ptrs, vals, mask=mask)


def _run_triton(final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    # Ensure inputs are on CUDA and have expected shapes
    assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
    rows = final_hidden_states.shape[0]
    hidden = final_hidden_states.shape[1]
    num_selected = expert_outputs.shape[0]

    # Output: initialize to zeros by cloning (bf16)
    output = final_hidden_states.clone()

    # Cast indices to int32 for Triton
    indices_i32 = token_indices.to(torch.int32)

    # Launch Triton kernel
    BLOCK_SIZE = 256
    grid = (num_selected,)
    scatter_add_rows_kernel[grid](output, expert_outputs, indices_i32, rows, hidden, num_selected, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2)

    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton path (GPU); keep host code minimal and avoid PyTorch ops
        if TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            return _run_triton(final_hidden_states, expert_outputs, token_indices)
        else:
            # CPU fallback (for environments without Triton/CUDA)
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output


def run(*args):
    return ModelNew()(*args)
