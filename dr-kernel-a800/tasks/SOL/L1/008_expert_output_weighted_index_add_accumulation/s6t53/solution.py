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
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


# Host-side wrapper for launching the Triton kernel
def triton_clone_bf16(src: torch.Tensor, dst: torch.Tensor):
    # Ensure tensors are on CUDA and have the same dtype
    assert src.is_cuda and dst.is_cuda, "Tensors must be on CUDA for Triton."
    assert src.dtype == torch.bfloat16 and dst.dtype == torch.bfloat16, "Expect bfloat16 tensors."
    N = src.numel()
    if dst.numel() != N:
        raise RuntimeError("dst must have the same number of elements as src.")
    # Choose block size and warps based on N
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
    linear_copy_kernel[grid](src, dst, N, BLOCK=BLOCK, num_warps=num_warps)
    return dst


class ModelNew(torch.nn.Module):
    def forward(self, final_hidden_states: torch.Tensor, expert_outputs: torch.Tensor, token_indices: torch.Tensor):
        # Ensure all tensors are on the same device
        device = final_hidden_states.device
        if expert_outputs.device != device:
            expert_outputs = expert_outputs.to(device)
        if token_indices.device != device:
            token_indices = token_indices.to(device)

        # Output must be a clone of final_hidden_states (Triton copy), not modifying final_hidden_states
        output = torch.empty_like(final_hidden_states, device=device, dtype=final_hidden_states.dtype)
        triton_clone_bf16(final_hidden_states, output)

        # Accumulate expert outputs into output using token_indices along dim=0
        output.index_add_(0, token_indices, expert_outputs)
        return output


def run(*args):
    return ModelNew()(*args)
