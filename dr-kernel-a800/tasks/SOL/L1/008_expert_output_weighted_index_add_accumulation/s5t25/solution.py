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
    pid = tl.program_id(0)
    if pid >= num_selected:
        return

    idx = tl.load(indices_ptr + pid)  # int32
    if idx < 0 or idx >= rows:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    num_chunks = (hidden + BLOCK_SIZE - 1) // BLOCK_SIZE

    for chunk in range(0, num_chunks):
        start = chunk * BLOCK_SIZE
        col = start + offs
        mask = col < hidden

        exp_ptr = expert_ptr + pid * hidden + col
        expert_vec = tl.load(exp_ptr, mask=mask, other=0.0)

        out_ptr = output_ptr + idx * hidden + col
        tl.atomic_add(out_ptr, expert_vec, mask=mask)


def _triton_scatter_add_rows(output: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
    """
    Launch Triton kernel to perform scatter-add:
        output[token_indices[i]] += expert_outputs[i] for all i
    Preconditions:
      - output is (rows, hidden), dtype bfloat16, CUDA
      - expert_outputs is (num_selected_tokens, hidden), dtype bfloat16, CUDA
      - token_indices is (num_selected_tokens,), dtype int32, CUDA
    """
    assert output.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda
    assert output.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16

    rows, hidden = output.shape
    num_selected = expert_outputs.shape[0]

    grid = (num_selected,)
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


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized scatter-add:
            output = final_hidden_states.clone()
            output[token_indices[i]] += expert_outputs[i] for all i
        Returns updated output tensor.
        """
        if TRITON_AVAILABLE and final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda:
            output = final_hidden_states.contiguous()
            if token_indices.dtype != torch.int32:
                token_indices_i32 = token_indices.to(torch.int32)
            else:
                token_indices_i32 = token_indices
            _triton_scatter_add_rows(output, expert_outputs, token_indices_i32)
            return output
        else:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
            return output


def run(*args):
    return ModelNew()(*args)
