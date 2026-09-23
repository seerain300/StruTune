import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic_kernel(vals_ptr, N, out_ptr, BLOCK: tl.constexpr):
    """
    Each program handles BLOCK elements. For each value in [0, 255], it performs a single
    atomic_add to out_ptr[value]. This avoids dynamic loops and is robust.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(vals_ptr + offs, mask=mask, other=0)  # int32 values
    # One atomic per lane
    tl.atomic_add(out_ptr + vals, 1, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Flatten topk_idx to 1D and compute histogram in Triton.
        - Compute inclusive expert_offsets using torch.cumsum on device (counts are already on device).
        - Stable sort of flattened indices is done in PyTorch (data-independent on num_experts).
        Returns:
          sorted_token_indices: permutation indices (int32), length = topk_idx.numel()
          expert_offsets: int32 tensor of shape (num_experts+1,)
        """
        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.view(-1)
        N = flat.numel()
        # Cast to int32 for Triton (topk_idx values are expert IDs)
        vals = flat.to(torch.int32)

        # Histogram counts per expert id (0..255)
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel: one atomic per element
        BLOCK = 1024  # number of elements per program
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](vals, N, counts, BLOCK=BLOCK, num_warps=4)

        # Inclusive offsets: prefix sum of counts via torch.cumsum (on GPU)
        # expert_offsets[i+1] = sum of counts[0..i], with expert_offsets[0] = 0
        expert_offsets = torch.cumsum(counts, dim=0).to(torch.int32)
        # Pad with one extra element for the final split
        expert_offsets = torch.nn.functional.pad(expert_offsets, (1, 0), mode='constant', value=0)

        # Stable sort of flattened indices (PyTorch, data-independent on num_experts)
        sorted_token_indices = vals.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
