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
    # Each program handles one selected token index
    pid = tl.program_id(axis=0)

    # Bounds check for safety (in case grid > num_selected)
    if pid >= num_selected:
        return

    # Load the target row index for this program
    row_index = tl.load(indices_ptr + pid)

    # Bounds check for row index
    if row_index < 0 or row_index >= rows:
        return

    # Iterate over hidden dimension in chunks
    # Note: static_range allows Triton to unroll with compile-time constant
    for start in tl.static_range(0, hidden, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden

        # Load the chunk of expert_output for this index
        # pointer arithmetic: expert_outputs[pid, offs]
        ep = expert_ptr + pid * hidden + offs

        # Load a vector of that chunk; masked elements use 0.0 (bf16)
        expert_chunk = tl.load(ep, mask=mask, other=tl.zeros([BLOCK_SIZE], dtype=tl.bfloat16))

        # Compute output pointer for the corresponding row chunk
        op = output_ptr + row_index * hidden + offs

        # Atomic add the chunk into the output row
        tl.atomic_add(op, expert_chunk, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # Minimal host-side work: clone output, ensure dtypes/devices, and launch Triton kernel.
        # Output must be a new tensor (not modifying input).
        output = final_hidden_states.clone()

        # Triton requires CUDA tensors; if not available, fall back to PyTorch implementation (for robustness).
        if not TRITON_AVAILABLE or output.device.type != "cuda" or expert_outputs.device.type != "cuda":
            # Fallback: PyTorch scatter-add, preserves correctness.
            result = output.clone()
            result.index_add_(dim=0, index=token_indices.to(result.device), source=expert_outputs)
            return result

        # Ensure dtypes: use bfloat16 for outputs/experts; indices must be int32 for Triton
        # The provided inputs are bfloat16; keep that. If not on CUDA, we already fallback.
        # Cast indices to int32 for Triton kernel
        indices_i32 = token_indices.to(torch.int32)

        # Launch configuration: one program per selected token
        grid = (expert_outputs.shape[0],)

        # num_warps/num_stages chosen for robust performance across varied hidden sizes
        # Keep BLOCK_SIZE=256 (compile-time constant) for good vectorization.
        scatter_add_rows_kernel[grid](
            output,               # output_ptr
            expert_outputs,       # expert_ptr
            indices_i32,          # indices_ptr
            output.shape[0],      # rows
            expert_outputs.shape[1],  # hidden
            expert_outputs.shape[0],  # num_selected
            BLOCK_SIZE=256,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
