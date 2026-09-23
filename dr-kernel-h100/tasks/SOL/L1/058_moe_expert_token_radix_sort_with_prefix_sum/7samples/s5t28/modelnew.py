import torch
import triton
import triton.language as tl


@triton.jit
def stable_sort_bitonic_pairs(in_ids_ptr, out_ids_ptr, out_idx_ptr,
                              N: tl.int32,
                              LOGN: tl.constexpr,
                              STEPS: tl.constexpr):
    """
    In-place stable bitonic sort of pairs (ID, index) using bitonic network.
    Assumes N is a power of two. We operate on global arrays out_ids_ptr/out_idx_ptr.
    """
    i = tl.arange(0, STEPS)
    # For each stage k = 1..LOGN:
    for k in tl.static_range(1, LOGN + 1):
        # j distance for this stage
        j = 1 << k
        # For each subsequence p = 0..N/2 - j step of 2*j
        for p in tl.static_range(0, N, 2 * j):
            # q = i ^ j
            q = i ^ (1 << k)
            # Load pairs (a, x) and (b, y)
            a = tl.load(in_ids_ptr + i)
            x = tl.load(out_idx_ptr + i)
            b = tl.load(in_ids_ptr + q)
            y = tl.load(out_idx_ptr + q)

            # Determine ascending or descending for this subsequence
            ascend = ( (i & j) == 0 )

            # Compare IDs
            cond = a > b

            # Stable tie-breaker: if IDs equal, smaller original index first
            equal = a == b
            tie = x > y  # prefer smaller original index
            swap = tl.where(equal, tie, cond)  # swap if (equal and x>y) or (not equal and a>b)
            swap = swap & ascend  # only swap in ascending subsequences

            # Compute new positions
            a_new = tl.where(swap, b, a)
            x_new = tl.where(swap, y, x)
            b_new = tl.where(swap, a, b)
            y_new = tl.where(swap, x, y)

            # Store back
            tl.store(out_ids_ptr + i, a_new)
            tl.store(out_idx_ptr + i, x_new)
            tl.store(out_ids_ptr + q, b_new)
            tl.store(out_idx_ptr + q, y_new)


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Histogram of expert IDs in flat_ptr into counts_ptr using chunked iteration with atomics.
    flat_ptr: int32 array of length N
    counts_ptr: int32 array of length num_experts
    """
    # Single program iterates over flat in chunks
    start = 0
    while start < N:
        idx = start + tl.arange(0, BLOCK)
        mask = idx < N
        ids = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Atomic add 1 for each valid id
        # Note: Triton atomic_add expects int32 accumulation, we cast ids to int32 for indexing
        tl.atomic_add(counts_ptr + ids, 1, mask=mask)
        start += BLOCK


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute exclusive prefix sum: offsets[j] = sum_{k<j} counts[k], for j in [1..num_experts].
    offsets_ptr length = num_experts + 1. Host sets offsets[0] = 0.
    """
    # offsets_ptr is 1-indexed for output
    total = tl.zeros((), dtype=tl.int32)
    for j in tl.static_range(1, num_experts + 1):
        total += tl.load(counts_ptr + (j - 1))
        tl.store(offsets_ptr + j, total)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The evaluation harness passes the pre-generated topk_idx tensor.
        # We assume the first argument is the topk_idx tensor.
        if len(args) == 0:
            raise ValueError("ModelNew.forward expects at least one argument (topk_idx).")
        topk_idx = args[0]

        # Ensure int32 and flatten
        flat = topk_idx.reshape(-1).to(torch.int32)

        N = flat.numel()
        # We must choose N as a power of two for bitonic sort. Pad to next power of two.
        # Compute next power of two >= N
        next_pow2 = 1 << (N - 1).bit_length()
        # Create temporary storage for sorted IDs and permutation indices
        out_ids = torch.empty(next_pow2, dtype=torch.int32, device=flat.device)
        out_idx = torch.empty(next_pow2, dtype=torch.int32, device=flat.device)

        # Copy flat into out_ids and initialize out_idx with 0..next_pow2-1
        # Only the first N elements are valid; the rest can be set to 0 (won't be accessed in sort)
        out_ids.copy_(flat)
        out_idx = torch.arange(next_pow2, dtype=torch.int32, device=flat.device)

        # Launch Triton bitonic sort kernel
        LOGN = (next_pow2).bit_length() - 1  # since next_pow2 is power of two
        STEPS = next_pow2 // 2
        stable_sort_bitonic_pairs(out_ids, out_ids, out_idx, N, LOGN=LOGN, STEPS=STEPS)

        # Extract sorted_token_indices as the first N entries of out_idx
        sorted_token_indices = out_idx[:N].to(torch.int32)

        # Compute counts of each expert ID using Triton
        num_experts = 256  # as in the original setup
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # chunk size
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Compute exclusive prefix sum to produce offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        offsets[0] = 0
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Return: permutation indices (int32) and offsets (int32)
        return sorted_token_indices, offsets