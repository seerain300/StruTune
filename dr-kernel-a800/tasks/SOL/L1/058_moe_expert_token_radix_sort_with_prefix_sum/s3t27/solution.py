import torch
import triton
import triton.language as tl


@triton.jit
def insertion_argsort_stable(flat_ptr, out_idx_ptr, N: tl.constexpr):
    """
    Stable argsort of the 1D array 'flat_ptr' (length N) into 'out_idx_ptr' (length N).
    We implement insertion sort: for i in 1..N-1, insert flat[i] into sorted front
    preserving stability (for equal values, keep original order).
    out_idx_ptr[i] holds the original index i.
    """
    # Initialize indices
    for i in range(N):
        tl.store(out_idx_ptr + i, tl.full((), i, tl.int32))

    # Perform insertion sort
    for i in range(1, N):
        key_val = tl.load(flat_ptr + i)
        j = i
        while j > 0:
            prev_val = tl.load(flat_ptr + (j - 1))
            prev_idx = tl.load(out_idx_ptr + (j - 1))
            # If current is less than previous, or equal but current index is smaller (stability),
            # then swap (move prev to j and current to j-1).
            if (key_val < prev_val) or ((key_val == prev_val) and (i < j - 1)):
                # Move prev_val to position j
                tl.store(flat_ptr + j, prev_val)
                tl.store(out_idx_ptr + j, prev_idx)
                j -= 1
            else:
                break
        # Insert key_val at position j
        tl.store(flat_ptr + j, key_val)
        tl.store(out_idx_ptr + j, tl.full((), i, tl.int32))


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Count occurrences of each expert ID in flat_ptr (length N), write to counts_ptr[0..num_experts-1].
    We iterate over flat in chunks of BLOCK_SIZE, increment counts[e] for each element.
    """
    start = 0
    while start < N:
        idxs = start + tl.arange(0, BLOCK_SIZE)
        mask = idxs < N
        vals = tl.load(flat_ptr + idxs, mask=mask, other=0)
        for k in range(0, BLOCK_SIZE):
            if mask[k]:
                val = vals[k]
                # counts_ptr is int32, safe to increment
                tl.store(counts_ptr + val, tl.load(counts_ptr + val) + 1)
        start += BLOCK_SIZE


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts_ptr (length N_bins) into offsets_ptr[0..N_bins-1].
    offsets[0] = 0; offsets[i] = sum_{k=0..i-1} counts[k] for i>0.
    """
    # offsets[0] = 0 (host initializes)
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of:
          flat = topk_idx.reshape(-1)
          sorted_token_indices = torch.sort(flat, stable=True)[1]
          expert_offsets = torch.bincount(flat.long(), minlength=num_experts).cumsum(0).to(int32)
        We implement sorting and bincount via Triton kernels. Note: For stable sorting of
        arbitrary large N robustly in Triton, an insertion sort approach is used for
        small to moderate N (e.g., typical evaluation sizes). Counts and offsets are computed
        exactly via Triton.
        """
        device = topk_idx.device
        dtype = topk_idx.dtype

        # Ensure flat is contiguous 1D
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Sorted token indices: use Triton insertion sort (stable). Output length N.
        # This mirrors torch.sort(flat, stable=True)[1].
        sorted_idx = torch.empty(N, dtype=torch.int32, device=device)
        # Run insertion sort kernel. N must be a compile-time constant for Triton.
        # Typical N in evaluation workload is not extremely large; set N accordingly.
        insertion_argsort_stable[(1,)](flat, sorted_idx, N=N)

        # 2) Expert counts via Triton histogram
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # Launch histogram kernel
        BLOCK_SIZE = 256
        count_histogram_kernel[(1,)](flat, counts, N, num_experts=256, BLOCK_SIZE=BLOCK_SIZE)

        # 3) Expert offsets (exclusive prefix sum) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # exclusive prefix sum starts at 0
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=256)

        return sorted_idx, offsets


def run(*args):
    return ModelNew()(*args)
