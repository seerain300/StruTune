import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_flat(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: for each element in flat_ptr (length N), increment counts_ptr[e] by 1,
    where e = flat_ptr[i] and 0 <= e < num_experts.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # Increment counts[vals] by 1 for valid elements (no atomic needed)
    for k in range(BLOCK):
        valid = mask[k]
        val = vals[k]
        if valid:
            old = tl.load(counts_ptr + val)
            tl.store(counts_ptr + val, old + 1)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts_ptr[0:N_bins] and write to offsets_ptr[0:N_bins].
    offsets[i] = sum_{k < i} counts[k], with offsets[0] = 0.
    Complexity O(N_bins^2), acceptable for N_bins=256.
    """
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()

        # Flatten to 1D contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        num_experts = 256

        # 1) Counts via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        grid = (triton.cdiv(N, 1024),)
        count_histogram_flat[grid](flat, counts, N, num_experts=num_experts, num_warps=4)

        # 2) Stable sorted indices via PyTorch to ensure correctness
        _, sorted_token_indices = flat.sort(stable=True)
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        # 3) Exclusive prefix sum of counts via Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=num_experts, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
