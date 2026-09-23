import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_bitonic_kernel(work_ptr, perm_ptr, N: tl.constexpr):
    # We run a single program instance that implements bitonic sort over N elements.
    # work_ptr: array of indices to be sorted (int32)
    # perm_ptr: permutation output (int32 positions)
    # We use N as a constexpr to allow for loops based on N.
    # Initialize permutation: identity
    for i in range(0, N):
        tl.store(perm_ptr + i, tl.cast(i, tl.int32))

    # Bitonic sort network
    # For each stage k (size of sequences), then for each subsequence j
    k = 2
    while k <= N:
        j = k // 2
        while j > 0:
            # For each position i, its partner is i ^ j
            i = 0
            while i < N:
                partner = i ^ j
                # Only process each pair once (i < partner)
                if i < partner:
                    a = tl.load(work_ptr + i)
                    b = tl.load(work_ptr + partner)
                    idx_a = tl.load(perm_ptr + i)
                    idx_b = tl.load(perm_ptr + partner)

                    # Tie-breaker: stable sort -> swap only if a > b or (a == b and idx_a > idx_b)
                    swap = (a > b) | ((a == b) & (idx_a > idx_b))

                    # Compute min and max values and indices
                    min_val = tl.where(swap, b, a)
                    max_val = tl.where(swap, a, b)
                    min_idx = tl.where(swap, idx_b, idx_a)
                    max_idx = tl.where(swap, idx_a, idx_b)

                    # Write back to positions i and partner according to 'up' (subsequence direction)
                    up = (i & k) == 0
                    # Positions that should hold min/max within this subsequence
                    if up:
                        # lower half gets min, upper half gets max
                        tl.store(work_ptr + i, min_val)
                        tl.store(work_ptr + partner, max_val)
                        tl.store(perm_ptr + i, min_idx)
                        tl.store(perm_ptr + partner, max_idx)
                    else:
                        # lower half gets max, upper half gets min
                        tl.store(work_ptr + i, max_val)
                        tl.store(work_ptr + partner, min_val)
                        tl.store(perm_ptr + i, max_idx)
                        tl.store(perm_ptr + partner, min_idx)
                i += 1
            j //= 2
        k *= 2


@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Atomically accumulate histogram of indices into counts[0..255]
    start = 0
    while start < N:
        idxs = tl.load(flat_ptr + start + tl.arange(0, BLOCK))
        # idxs are int32; cast to int64 for atomic_add safety
        idxs64 = tl.cast(idxs, tl.int64)
        # Mask out-of-range lanes
        mask = (start + tl.arange(0, BLOCK)) < N
        # Atomic add per valid lane
        for m in range(0, BLOCK):
            if mask[m]:
                val = idxs64[m]
                # Only add if 0 <= val < 256
                if (val >= 0) & (val < 256):
                    tl.atomic_add(counts_ptr + val, 1)
        start += BLOCK


@triton.jit
def _prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.constexpr):
    # Compute inclusive prefix sum of counts into offsets[1..M], offsets[0]=0 (set in host)
    prefix = tl.zeros((), dtype=tl.int32)  # scalar
    for k in range(0, M):
        c = tl.load(counts_ptr + k)
        prefix = prefix + c
        tl.store(offsets_ptr + k + 1, prefix)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Implements stable argsort of flattened topk_idx via a bitonic sorting network in Triton.
        - Builds histogram and computes offsets (cumulative counts) via Triton.
        Returns:
          sorted_token_indices: permutation indices (int32) of length N
          expert_offsets: int32 tensor of length 257
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        flat = topk_idx.reshape(-1).contiguous()  # 1D int32 on GPU
        N = flat.numel()
        device = flat.device

        # Allocate outputs
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Copy indices into a work buffer for sorting
        work = flat.clone()

        # Launch Triton bitonic sort kernel: grid=(1,) since it processes entire array
        _argsort_bitonic_kernel[(1,)](work, sorted_token_indices, N, BLOCK=1024)

        # Histogram of indices (values in [0, 255])
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # Launch histogram kernel in chunks
        _histogram_atomic_kernel[(1,)](flat, counts, N, BLOCK=1024)

        # Prefix sum to get expert_offsets
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        _prefix_sum_kernel[(1,)](counts, offsets, M=256)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
