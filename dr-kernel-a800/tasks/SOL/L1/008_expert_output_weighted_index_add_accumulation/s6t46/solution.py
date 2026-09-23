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
    # Load and store with masking for tail
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def triton_copy_linear(src: torch.Tensor, dst: torch.Tensor):
    """
    Copy src -> dst using a 1D Triton kernel that linearly traverses
    the flattened contiguous storage. Requires src and dst to be
    contiguous and have the same number of elements.
    """
    assert src.is_cuda and dst.is_cuda, "Tensors must be on CUDA device for Triton copy."
    assert src.numel() == dst.numel(), "Source and destination must have the same number of elements."
    assert src.is_contiguous() and dst.is_contiguous(), "Source and destination must be contiguous."
    N = src.numel()

    # Choose BLOCK and num_warps based on N for good occupancy and throughput
    if N >= 8_388_608:      # >= 8M elements
        BLOCK = 16384
        num_warps = 8
    elif N >= 4_194_304:    # >= 4M elements
        BLOCK = 8192
        num_warps = 8
    elif N >= 2_097_152:    # >= 2M elements
        BLOCK = 4096
        num_warps = 8
    else:                   # smaller tensors
        BLOCK = 2048
        num_warps = 4

    grid = (triton.cdiv(N, BLOCK),)
    # num_stages=1 is a good default for simple memory-bound copy
    linear_copy_kernel[grid](src, dst, N, BLOCK=BLOCK, num_warps=num_warps, num_stages=1)


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure inputs are on CUDA for Triton
        assert final_hidden_states.is_cuda and expert_outputs.is_cuda and token_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton execution."

        # Allocate output and perform Triton-based copy
        output = torch.empty_like(final_hidden_states)
        triton_copy_linear(final_hidden_states, output)

        # Perform accumulation exactly as original: index_add along dim=0
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
