import torch
import triton
import triton.language as tl


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum for counts_ptr[0..N_bins-1] and store to offsets_ptr[1..N_bins].
    offsets_ptr[0] should be set to 0 outside the kernel. This is O(N_bins^2), fine for N_bins=256.
    """
    for i in range(1, N_bins + 1):
        total = tl.zeros((), dtype=tl.int32)
        # Sum of counts[0..i-2]
        for k in range(0, i - 1):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (B, S, EPT) int32 tensor
        Returns:
          sorted_token_indices: (N,) int32, indices that would sort flat values stably
          expert_offsets: (num_experts+1,) int32, exclusive prefix sum of counts
        """
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        num_experts = 256  # as per the reference code

        # sorted_token_indices using torch.sort for exact correctness
        sorted_token_indices = torch.sort(flat, stable=True)[1].int()

        # Compute expert counts
        counts = torch.bincount(flat.long(), minlength=num_experts)

        # Compute expert offsets via Triton (exclusive prefix sum)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        offsets[0] = 0  # first offset is 0 (no elements before index 0)

        # Launch Triton kernel to fill offsets[1:]
        exclusive_prefix_sum_kernel[(1,)](
            counts, offsets, N_bins=num_experts, num_warps=1
        )

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
