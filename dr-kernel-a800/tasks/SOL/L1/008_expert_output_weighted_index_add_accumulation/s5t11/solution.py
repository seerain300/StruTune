import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: scatter-add expert_outputs[i, :] into output[token_indices[i], :]
# One program per selected token. Vectorize across hidden dimension in chunks of BLOCK_SIZE.
@triton.jit
def scatter_add_rows_kernel(
    output_ptr,         # *ptr to output [rows, hidden]
    expert_ptr,         # *ptr to expert_outputs [num_selected_tokens, hidden]
    indices_ptr,        # *ptr to token_indices [num_selected_tokens] (int32)
    rows,               # number of rows in output (batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # num_selected_tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension
):
    pid = tl.program_id(axis=0)  # which expert output row to process
    if pid >= num_selected:
        return

    # Load target row index (int32)
    idx = tl.load(indices_ptr + pid)

    # Iterate over hidden dimension in chunks of BLOCK_SIZE
    for j in range(0, hidden, BLOCK_SIZE):
        offs = j + tl.arange(0, BLOCK_SIZE)        # vector of column offsets
        mask = offs < hidden                       # mask for tail

        # Load expert chunk: expert_ptr is row-major; offset = pid * hidden + offs
        exp = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)

        # Compute output addresses for this row chunk: out[row, col] -> row * hidden + col
        out_addrs = idx * hidden + offs

        # Atomic add into output
        tl.atomic_add(output_ptr + out_addrs, exp, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # If Triton is not available, fallback to original PyTorch behavior for correctness
        if not TRITON_AVAILABLE:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output

        # Ensure inputs are CUDA tensors
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Inputs must be CUDA tensors."

        # Clone to match original behavior
        output = final_hidden_states.clone()

        # Cast indices to int32 for Triton (kernel expects int32)
        if token_indices.dtype != torch.int32:
            token_indices = token_indices.to(torch.int32)

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Launch Triton kernel: one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output,
            expert_outputs,
            token_indices,
            rows,
            hidden,
            num_selected,
            BLOCK_SIZE=256,           # fixed chunk size, empirically fast and stable
            num_warps=4,              # balanced for memory-bound ops
            num_stages=2,             # pipelining
        )

        return output


def run(*args):
    return ModelNew()(*args)
