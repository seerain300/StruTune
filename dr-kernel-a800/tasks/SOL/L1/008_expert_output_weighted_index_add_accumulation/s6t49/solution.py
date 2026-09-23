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
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def triton_copy(final_hidden_states: torch.Tensor) -> torch.Tensor:
    # Allocate output and launch Triton copy kernel
    B, H = final_hidden_states.shape
    output = torch.empty_like(final_hidden_states)
    N = B * H

    # Choose BLOCK and num_warps based on N
    if N >= 16 * 1024 * 1024:
        BLOCK = 32768
        num_warps = 8
    elif N >= 4 * 1024 * 1024:
        BLOCK = 16384
        num_warps = 8
    elif N >= 1 * 1024 * 1024:
        BLOCK = 8192
        num_warps = 8
    else:
        BLOCK = 4096
        num_warps = 4

    grid = (triton.cdiv(N, BLOCK),)
    linear_copy_kernel[grid](
        final_hidden_states,
        output,
        N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on the same device and dtype
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, "Tensors must be on CUDA for Triton."
        # Triton copy to produce output starting with final_hidden_states
        output = triton_copy(final_hidden_states)
        # Accumulate expert outputs into corresponding token positions
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
