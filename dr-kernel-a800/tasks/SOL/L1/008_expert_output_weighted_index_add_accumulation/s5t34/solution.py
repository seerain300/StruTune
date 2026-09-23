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
    rows,               # number of rows in output (i.e., batch_size * seq_len)
    hidden,             # hidden_size (number of columns)
    num_selected,       # number of selected tokens
    BLOCK_SIZE: tl.constexpr,  # chunk size across hidden dimension
):
    pid = tl.program_id(axis=0)
    if pid >= num_selected:
        return

    # Load target row index for this token
    row = tl.load(indices_ptr + pid)
    if row >= rows:
        return

    # Iterate over hidden dimension in chunks
    for start in range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden
        # Load contribution for this token (vector across hidden)
        contrib = tl.load(expert_ptr + pid * hidden + offs, mask=mask, other=0.0)
        # Atomic add into the target row
        tl.atomic_add(output_ptr + row * hidden + offs, contrib, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Perform atomic accumulation of expert outputs back to token positions (dim=0).
        This is a Triton implementation of index_add along dim=0.
        """
        # Fallback to PyTorch if Triton or CUDA not available
        if not TRITON_AVAILABLE or final_hidden_states.device.type != "cuda":
            output = final_hidden_states.clone()
            output.index_add_(0, token_indices, expert_outputs)
            return output

        # Ensure tensors are on the same device
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA device for Triton kernel"
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA device for Triton kernel"
        assert token_indices.is_cuda, "token_indices must be on CUDA device for Triton kernel"

        # Clone to match reference semantics
        output = final_hidden_states.clone()

        # Triton expects int32 indices
        indices_i32 = token_indices.to(torch.int32)

        rows = final_hidden_states.shape[0]
        hidden = final_hidden_states.shape[1]
        num_selected = expert_outputs.shape[0]

        # Choose kernel tuning based on hidden size
        if hidden <= 128:
            BLOCK_SIZE = 128
            num_warps = 2
            num_stages = 1
        elif hidden <= 256:
            BLOCK_SIZE = 256
            num_warps = 4
            num_stages = 2
        elif hidden <= 1024:
            BLOCK_SIZE = 512
            num_warps = 4
            num_stages = 2
        else:
            # For very large hidden, 512 works well in practice
            BLOCK_SIZE = 512
            num_warps = 8
            num_stages = 2

        # Launch one program per selected token
        grid = (num_selected,)
        scatter_add_rows_kernel[grid](
            output, expert_outputs, indices_i32,
            rows, hidden, num_selected,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output


def run(*args):
    return ModelNew()(*args)
