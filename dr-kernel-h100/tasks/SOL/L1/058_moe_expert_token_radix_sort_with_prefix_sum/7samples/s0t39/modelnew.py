import torch
import triton
import triton.language as tl


@triton.jit
def triton_bincount_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Triton kernel to compute per-expert counts of values in flat_ptr (int32).
    We assume values are in [0, 255] (num_experts=256). We process BLOCK elements per program.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load flat values; out-of-range lanes get 0 but will be masked on store.
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # We only care about values in [0, 255]. For others, ignore by not storing.
    valid = (vals >= 0) & (vals <= 255) & mask
    # Atomic add 1 for each valid value
    # Note: counts_ptr is int32, vals are int32
    tl.atomic_add(counts_ptr + vals, 1, mask=valid)


@triton.jit
def triton_inclusive_prefix_sum_int64(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Triton kernel to compute inclusive prefix sum of counts (int32) into offsets (int64) of length L.
    We start at offsets[0] = 0, then offsets[i] = offsets[i-1] + counts[i-1] (counts[i-1] as int64).
    """
    # Single program performs the loop sequentially; L is constexpr.
    prev = tl.zeros((), dtype=tl.int64)
    for i in range(0, L):
        val = tl.load(counts_ptr + (i - 1))  # counts[i-1], with counts[-1] unused in inclusive sum
        # val is int32; cast to int64 for accumulation
        val64 = val.to(tl.int64)
        curr = prev + val64
        tl.store(offsets_ptr + i, curr)
        prev = curr


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Triton kernels: bincount of expert ids in [0, 255] (int32) and inclusive prefix sum (int64).
        - PyTorch: stable argsort for flattened indices to produce sorted_token_indices (int32).
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount: counts[0..255] as int32
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        triton_bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0  # inclusive, starting at 0
        triton_inclusive_prefix_sum_int64[(1,)](counts, offsets, L=257)

        # 3) PyTorch stable argsort for flattened indices (returns permutation of [0, N-1])
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets