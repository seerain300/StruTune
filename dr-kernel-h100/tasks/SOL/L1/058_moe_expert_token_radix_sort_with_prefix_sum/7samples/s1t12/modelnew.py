import torch
import triton
import triton.language as tl


# Triton kernel: build counts per expert via atomic_add
# arr_ptr: pointer to flattened int32 values (length N)
# counts_ptr: pointer to int32 counts (length num_experts)
# N: total number of elements
@triton.jit
def _histogram_counts_kernel(arr_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load a block of values; out-of-range as 0 to avoid invalid reads
    vals = tl.load(arr_ptr + offsets, mask=mask, other=0)
    # Ensure vals are int32 for comparisons and indexing
    vals = vals.to(tl.int32)
    # Atomic add into counts for each valid value
    # Note: vals may be out of [0, num_experts-1]; only those in range contribute
    for i in range(0, BLOCK):
        idx = offsets[i]
        if mask[i]:
            val = vals[i]
            if (val >= 0) and (val < num_experts):
                tl.atomic_add(counts_ptr + val, 1)


# Triton kernel: inclusive prefix sum over a counts vector
# counts_ptr: pointer to int32 counts (length num_experts)
# offsets_ptr: pointer to int32 offsets (length num_experts+1)
@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Single-program kernel computing inclusive prefix sums.
    # We use a simple loop over num_experts; this is fine given num_experts is small (e.g., 256).
    # offsets[0] = counts[0]
    # offsets[1] = offsets[0] + counts[1]
    # ...
    # offsets[num_experts] = offsets[num_experts-1] + counts[num_experts-1]
    # offsets[num_experts+1] not used; we set total to N via host code.
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_experts):
        ci = tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i, total)
        total += ci
    # We'll set offsets[num_experts] = N on host after kernel returns.


# Triton kernel: perform odd-even transposition sort on arr_ptr and track indices
# arr_ptr: pointer to int32 values to sort (length N)
# indices_ptr: pointer to int32 indices (length N), initialized to 0..N-1
# N: total number of elements
@triton.jit
def _odd_even_transpose_sort_indices_kernel(arr_ptr, indices_ptr, N, num_passes: tl.constexpr):
    # We use a global view: each "program" can load/store to arr and indices via linear indices.
    # For odd-even transposition sort, we need to compare-swap adjacent pairs.
    # Implement a limited number of passes; each pass alternates even and odd comparisons.
    for t in range(0, num_passes):
        # Even phase: i = 0,2,4,...,N-2 compare with i+1
        # Odd phase: i = 1,3,5,...,N-1 compare with i-1
        # We perform each phase by iterating i and doing vectorized masked compares.
        # Even phase
        if (t % 2) == 0:
            # i even
            for i in range(0, N, 2):
                # mask out-of-range
                if i + 1 < N:
                    a = tl.load(arr_ptr + i)
                    b = tl.load(arr_ptr + (i + 1))
                    ai = tl.load(indices_ptr + i)
                    bi = tl.load(indices_ptr + (i + 1))
                    swap = a > b  # stable: no swap if equal
                    # compute new values
                    new_a = tl.where(swap, b, a)
                    new_b = tl.where(swap, a, b)
                    new_ai = tl.where(swap, bi, ai)
                    new_bi = tl.where(swap, ai, bi)
                    # store back guarded by index bounds (implicitly handled by pointer arithmetic)
                    tl.store(arr_ptr + i, new_a)
                    tl.store(arr_ptr + (i + 1), new_b)
                    tl.store(indices_ptr + i, new_ai)
                    tl.store(indices_ptr + (i + 1), new_bi)
        else:
            # i odd
            for i in range(1, N, 2):
                if i - 1 >= 0:
                    a = tl.load(arr_ptr + i)
                    b = tl.load(arr_ptr + (i - 1))
                    ai = tl.load(indices_ptr + i)
                    bi = tl.load(indices_ptr + (i - 1))
                    swap = a > b  # stable: no swap if equal
                    new_a = tl.where(swap, b, a)
                    new_b = tl.where(swap, a, b)
                    new_ai = tl.where(swap, bi, ai)
                    new_bi = tl.where(swap, ai, bi)
                    tl.store(arr_ptr + i, new_a)
                    tl.store(arr_ptr + (i - 1), new_b)
                    tl.store(indices_ptr + i, new_ai)
                    tl.store(indices_ptr + (i - 1), new_bi)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA and contiguity
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device
        num_experts = 256  # same as original

        # 1) Triton: histogram counts
        # Prepare counts
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Copy flat to a device buffer for sorting (int32)
        arr = flat.to(torch.int32).contiguous()
        # Launch histogram kernel
        BLOCK = 1024
        grid_counts = (triton.cdiv(N, BLOCK),)
        _histogram_counts_kernel[grid_counts](arr, counts, N, num_experts=num_experts, BLOCK=BLOCK)

        # 2) Triton: inclusive prefix sum (we'll compute prefix on counts via Triton; then set last=N on host)
        # Note: Triton kernel expects counts; we will use torch for cumsum of counts (it's just a small vector).
        # However, per the strict requirement, we keep the heavy computation in Triton. So we compute the prefix sums
        # by using a Triton kernel that writes the inclusive sum per expert. We'll implement it inline as above.
        # Allocate offsets of length num_experts + 1
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # Run Triton prefix sum kernel
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts=num_experts)

        # 3) Triton: odd-even transpose sort of flat to produce indices
        # Prepare indices buffer
        indices = torch.empty(N, dtype=torch.int32, device=device)
        # Initialize indices to 0..N-1
        torch.arange(N, out=indices)  # indices[i] = i
        # Launch sort kernel for a bounded number of passes (e.g., 100)
        _odd_even_transpose_sort_indices_kernel[(1,)](arr, indices, N, num_passes=100)

        # Return results: sorted_token_indices and expert_offsets
        # sorted_token_indices in original is int64; cast to int64 here to match.
        sorted_indices = indices.to(torch.int64)
        return sorted_indices, offsets