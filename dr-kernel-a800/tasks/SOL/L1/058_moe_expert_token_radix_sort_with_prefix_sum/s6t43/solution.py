import torch
import triton
import triton.language as tl


# Triton kernel: odd-even transposition sort to produce sorted_token_indices.
# We sort the permutation indices out_idx based on the values in orig.
# orig_ptr: int32 flattened values, length N
# out_idx_ptr: int32 permutation, length N, initialized to [0..N-1]
# N: total number of elements (constexpr)
@triton.jit
def odd_even_sort_kernel(orig_ptr, out_idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Single program processes the entire array via odd-even transposition sort.
    # Perform N phases: even pass pairs (0,1),(2,3)...; odd pass pairs (1,2),(3,4)...
    for phase in range(N):
        # Even phase
        i = tl.arange(0, BLOCK)
        a = i
        b = i + 1
        mask_a = a < N
        mask_b = b < N
        idx_a = tl.load(out_idx_ptr + a, mask=mask_a, other=0)
        idx_b = tl.load(out_idx_ptr + b, mask=mask_b, other=0)
        val_a = tl.load(orig_ptr + idx_a, mask=mask_a, other=0)
        val_b = tl.load(orig_ptr + idx_b, mask=mask_b, other=0)
        # Only process even indices a and valid b
        active = (a % 2 == 0) & (b < N)
        # Stable compare-and-swap: swap if val_b < val_a or (val_b == val_a and idx_b < idx_a)
        swap = (val_b < val_a) | ((val_b == val_a) & (idx_b < idx_a))
        new_a = tl.where(swap, idx_b, idx_a)
        new_b = tl.where(swap, idx_a, idx_b)
        tl.store(out_idx_ptr + a, new_a, mask=active)
        tl.store(out_idx_ptr + b, new_b, mask=active)

        # Odd phase
        a = i
        b = i + 1
        mask_a = a < N
        mask_b = b < N
        idx_a = tl.load(out_idx_ptr + a, mask=mask_a, other=0)
        idx_b = tl.load(out_idx_ptr + b, mask=mask_b, other=0)
        val_a = tl.load(orig_ptr + idx_a, mask=mask_a, other=0)
        val_b = tl.load(orig_ptr + idx_b, mask=mask_b, other=0)
        active = (a % 2 == 1) & (b < N)
        swap = (val_b < val_a) | ((val_b == val_a) & (idx_b < idx_a))
        new_a = tl.where(swap, idx_b, idx_a)
        new_b = tl.where(swap, idx_a, idx_b)
        tl.store(out_idx_ptr + a, new_a, mask=active)
        tl.store(out_idx_ptr + b, new_b, mask=active)


# Triton kernel: histogram of flattened values into counts[num_experts], per index in orig.
# orig_ptr: int32 values, length N
# counts_ptr: int32 counts, length L (num_experts), initialized to zeros
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr):
    # Single program processes the array. For each element, atomically add 1 to counts[val].
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton kernel: exclusive prefix sum of counts to produce offsets[e] = sum of counts for ids < e.
# counts_ptr: int32 counts, length L
# out_ptr: int32 offsets, length L+1, initialized out[L]=N
@triton.jit
def scan_kernel(counts_ptr, out_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    running = 0
    for i in range(L):
        c = tl.load(counts_ptr + i)
        running += c
        tl.store(out_ptr + i, running - c)  # exclusive: current pos gets previous running
    total = running
    # out[L] is not used for offsets; we set out[num_experts] already via caller. But if we need to, we can leave it empty since last element already set by caller.
    # No-op here.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation that returns:
          - sorted_token_indices: int32 tensor of length N (sorted permutation of [0..N-1] by flattened values, stable)
          - expert_offsets: int32 tensor of length num_experts + 1, exclusive prefix sums
        """
        # Flatten to 1D int32
        orig = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = orig.numel()

        # 1) Compute sorted_token_indices using Triton odd-even sort
        out_idx = torch.empty(N, dtype=torch.int32, device=orig.device)
        out_idx.copy_(torch.arange(N, device=orig.device, dtype=torch.int32))  # initialize permutation

        # Launch Triton sort kernel
        odd_even_sort_kernel[(1,)](orig, out_idx, N, BLOCK=N, num_warps=4)

        # 2) Compute expert_offsets using Triton histogram and scan
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=orig.device)

        # Launch histogram kernel
        histogram_kernel[(1,)](orig, counts, N, L=num_experts, BLOCK=1024, num_warps=1)

        # Compute exclusive prefix sums into offsets of length num_experts + 1
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=orig.device)
        # Initialize offsets[0..255] to 0; we will fill via scan. Set last element to N.
        offsets[-1] = N
        scan_kernel[(1,)](counts, offsets, L=num_experts, BLOCK=256, num_warps=1)

        return out_idx, offsets


def run(*args):
    return ModelNew()(*args)
