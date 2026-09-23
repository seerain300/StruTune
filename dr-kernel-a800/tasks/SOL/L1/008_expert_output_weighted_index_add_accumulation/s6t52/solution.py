import torch
import triton
import triton.language as tl


@triton.jit
def linear_copy_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    N,        # int: total number of elements to copy
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load and store as bf16
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Triton copy: output must start as a clone of final_hidden_states
        output = torch.empty_like(final_hidden_states)
        N = output.numel()

        # Choose launch parameters based on N to balance occupancy and grid size
        if N >= 131072:
            BLOCK = 16384
            num_warps = 8
        elif N >= 16384:
            BLOCK = 8192
            num_warps = 8
        else:
            BLOCK = 4096
            num_warps = 4

        grid = (triton.cdiv(N, BLOCK),)
        linear_copy_kernel[grid](final_hidden_states, output, N, BLOCK=BLOCK, num_warps=num_warps, num_stages=1)

        # Accumulate expert outputs into output using PyTorch's optimized index_add
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
