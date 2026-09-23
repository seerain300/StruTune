import torch
import triton
import triton.language as tl


@triton.jit
def copy_1d_kernel(
    src_ptr,  # *const bfloat16
    dst_ptr,  # *bfloat16
    N,        # int: total number of elements to copy (batch_seq_len * hidden_size)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def triton_copy_1d(src: torch.Tensor, dst: torch.Tensor):
    """
    Copies src -> dst using a 1D Triton kernel over N = src.numel() elements.
    Assumes src and dst are contiguous and have the same shape/dtype.
    """
    assert src.is_cuda and dst.is_cuda, "Tensors must be on CUDA for Triton kernel."
    assert src.shape == dst.shape, "src and dst must have the same shape."
    assert src.dtype == dst.dtype, "src and dst must have the same dtype."
    assert src.is_contiguous() and dst.is_contiguous(), "src and dst must be contiguous."

    N = src.numel()
    # Heuristic tuning:
    # - Very large N: larger BLOCK and more warps/stages to saturate bandwidth.
    # - Moderate N: balanced BLOCK/warps/stages.
    if N >= 16_777_216:  # 16M elements and above
        BLOCK = 16384
        num_warps = 8
        num_stages = 3
    elif N >= 4_194_304:  # 4M to 16M
        BLOCK = 8192
        num_warps = 8
        num_stages = 2
    else:
        BLOCK = 4096
        num_warps = 4
        num_stages = 2

    grid = (triton.cdiv(N, BLOCK),)
    copy_1d_kernel[grid](src, dst, N, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    return dst


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        device = final_hidden_states.device
        assert final_hidden_states.is_cuda, "final_hidden_states must be on CUDA for Triton."
        assert expert_outputs.is_cuda, "expert_outputs must be on CUDA for Triton."
        assert token_indices.is_cuda, "token_indices must be on CUDA for Triton."
        assert final_hidden_states.dtype == torch.bfloat16 and expert_outputs.dtype == torch.bfloat16, "dtype must be bfloat16."

        # Create output and perform Triton-based copy
        output = torch.empty_like(final_hidden_states)
        triton_copy_1d(final_hidden_states, output)

        # Accumulation using PyTorch (preserves original semantics)
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
