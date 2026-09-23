import torch
import triton
import triton.language as tl


@triton.jit
def _row_add_atomic_kernel(
    out_ptr,         # *half, output buffer (final_hidden_states clone)
    expert_ptr,      # *half, expert_outputs (num_selected_tokens, hidden_size)
    indices_ptr,     # *int32, token_indices (num_selected_tokens,)
    N,               # int32, number of selected tokens
    H,               # int32, hidden size
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Destination row index for this selected token
    idx = tl.load(indices_ptr + pid)

    # Vector of hidden columns (process entire row)
    cols = tl.arange(0, BLOCK)
    mask = cols < H  # in case BLOCK > H (unused lanes masked)

    # Load source row values: expert_outputs[pid, cols]
    src_ptrs = expert_ptr + pid * H + cols
    vals = tl.load(src_ptrs, mask=mask, other=0.0)

    # Destination pointers for the row idx: out[idx, cols]
    dst_ptrs = out_ptr + idx * H + cols

    # Atomically add into the output
    tl.atomic_add(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        """
        Triton-optimized replacement for:
            output = final_hidden_states.clone()
            output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        """
        # Ensure contiguity
        final_hidden_states = final_hidden_states.contiguous()
        expert_outputs = expert_outputs.contiguous()
        token_indices = token_indices.contiguous()

        # Clone the accumulation buffer to avoid in-place modification
        output = final_hidden_states.clone()

        # Shapes
        N, H = expert_outputs.shape
        grid = (N,)

        # We'll set BLOCK to H so the kernel processes the entire hidden dimension at once.
        # Triton requires tl.constexpr for BLOCK; passing H (int) works as a compile-time constant per launch.
        BLOCK = int(H)

        # Launch kernel
        _row_add_atomic_kernel[grid](
            output, expert_outputs, token_indices,
            N, H,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )

        return output


def run(*args):
    return ModelNew()(*args)
