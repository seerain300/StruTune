import torch
import triton
import triton.language as tl


@triton.jit
def inclusive_cumsum_offsets_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Compute inclusive prefix sum of counts into offsets[1..]
    # Loop is simple and constexpr-sized, safe for num_experts=256.
    running = 0
    for i in range(num_experts):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)

        # Compute histogram with torch (fast and robust)
        num_experts = 256  # matches the original setup
        counts = torch.bincount(flat.long(), minlength=num_experts)  # int64 by default, fine for sums

        # Prepare offsets on device
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

        # Kernel: inclusive prefix sum for offsets[1..], offsets[0] = 0
        inclusive_cumsum_offsets_kernel[(1,)](counts, offsets, num_experts)
        # Set offsets[0] = 0
        offsets[0] = 0

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
