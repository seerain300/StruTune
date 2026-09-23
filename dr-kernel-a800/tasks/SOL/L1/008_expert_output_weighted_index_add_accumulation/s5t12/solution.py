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
    if pid >= num_selected:
        return

    # load target row index (int32)
    idx = tl.load(indices_ptr + pid)

    # iterate over hidden dimension in chunks
    # Note: we do pointer arithmetic in elements (not bytes)
    for j in range(0, hidden, BLOCK_SIZE):
        offs = j + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # load expert chunk: shape (BLOCK_SIZE,)
        expert_chunk = tl.load(
            expert_ptr + pid * hidden + offs,
            mask=mask,
            other=0.0
        )

        # compute output pointers for the target row
        out_ptrs = output_ptr + idx * hidden + offs

        # atomic add chunk into output
        tl.atomic_add(out_ptrs, expert_chunk, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Host-only preparation: clone to match original behavior and cast indices to int32 for Triton
        output = final_hidden_states.clone()
        # Ensure tensors are on the same device and dtype
        assert expert_outputs.device == output.device and token_indices.device == output.device, \
            "All tensors must be on the same device"
        # Triton expects int32 for indices
        indices_i32 = token_indices.to(torch.int32)

        rows = output.shape[0]
        hidden = output.shape[1]
        num_selected = expert_outputs.shape[0]

        # Choose launch config based on hidden size
        # BLOCK_SIZE=256 (works well for typical hidden sizes like 128/256/512/1024)
        BLOCK_SIZE = 256
        # Dynamically select num_warps and num_stages
        if hidden >= 512:
            num_warps = 8
        elif hidden >= 256:
            num_warps = 4
        else:
            num_warps = 2
        num_stages = 2

        # Launch Triton kernel
        if TRITON_AVAILABLE and output.is_cuda and expert_outputs.is_cuda and indices_i32.is_cuda:
            grid = (num_selected,)
            scatter_add_rows_kernel[grid](
                output, expert_outputs, indices_i32,
                rows, hidden, num_selected,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            # Fallback: pure PyTorch index_add for correctness if Triton not available or not CUDA
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)

        return output


def run(*args):
    return ModelNew()(*args)
