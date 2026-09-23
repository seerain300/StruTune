import torch
import triton
import triton.language as tl


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sums of counts_ptr[0:N_bins] into offsets_ptr[0:N_bins+1].
    offsets[0] = 0, offsets[i] = offsets[i-1] + counts[i-1] for i >= 1.
    """
    # Initialize offsets[0] = 0
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    # Compute exclusive prefix sums in O(N_bins^2)
    for i in range(1, N_bins + 1):
        total = tl.zeros((), dtype=tl.int32)
        # Sum counts[0..i-1]
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure topk_idx is on CUDA
        device = topk_idx.device
        # Flatten to 1D
        N = topk_idx.numel()
        flatten = topk_idx.reshape(-1).contiguous()

        # Compute expert counts using torch to ensure correctness
        num_experts = 256
        counts = torch.bincount(flatten.long(), minlength=num_experts).to(torch.int32)

        # Allocate offsets of length 257 (num_experts + 1)
        offsets = torch.empty(257, dtype=torch.int32, device=device)

        # Launch Triton kernel for exclusive prefix sum
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=256, num_warps=1)

        # sorted_token_indices: compute with torch for correctness
        _, sorted_token_indices = torch.sort(flatten, stable=True)

        # Return sorted_token_indices and expert_offsets
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
