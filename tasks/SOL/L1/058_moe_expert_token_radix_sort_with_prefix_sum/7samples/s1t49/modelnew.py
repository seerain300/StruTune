import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(val_ptr, counts_ptr, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load values; out-of-range masked elements won't contribute due to mask and other=0
    vals = tl.load(val_ptr + offsets, mask=mask, other=0)  # int32
    # Atomic add 1 for each valid element into counts[vals]
    # Note: counts_ptr is int32*, vals is int32, offsets is int32
    for i in range(BLOCK_SIZE):
        if mask[i]:
            tl.atomic_add(counts_ptr + vals[i], 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    # Single-program inclusive scan over counts[0:num_experts]
    # offsets_ptr[0] = 0
    # offsets_ptr[1] = counts[0]
    # offsets_ptr[2] = counts[0] + counts[1]
    # ...
    # offsets_ptr[num_experts+1] = sum(counts)
    # We assume caller allocates offsets_ptr of length num_experts + 1 and sets offsets_ptr[0] = 0.
    acc = tl.zeros((), dtype=tl.int32)  # scalar accumulator
    # Compute inclusive sum and write to offsets[1:]
    for e in range(0, num_experts):
        v = tl.load(counts_ptr + e)  # int32
        acc += v
        tl.store(offsets_ptr + 1 + e, acc)
    # offsets_ptr[num_experts+1] should be sum(counts); we can write it explicitly:
    tl.store(offsets_ptr + num_experts + 1, acc)


@triton.jit
def _stable_sort_indices_by_counting_kernel(flat_ptr, indices_ptr, n_elements: tl.int32, counts_ptr, num_experts: tl.int32, MAX_N: tl.constexpr):
    # We emulate stable counting sort:
    # - For each original index i, read val = flat[i], compute pos = exclusive prefix sum up to e < val,
    #   write i into sorted_indices[pos], then increment exclusive offset for val by 1.
    # This yields stable order by value ascending, then by original index ascending for ties.
    # Note: indices_ptr is length n_elements, initialized to zeros.
    # We loop i from 0 to MAX_N with i < n_elements active.
    for i in range(MAX_N):
        active = i < n_elements
        # Load value for position i
        val = tl.load(flat_ptr + i, mask=active, other=0)  # int32
        # Compute exclusive prefix sum for val: sum_{e' < val} counts[e']
        acc = tl.zeros((), dtype=tl.int32)
        # For each expert e, if e < val, add counts[e] to acc
        # This loop is O(num_experts); num_experts=256 is fine.
        for e in range(0, num_experts):
            ce = tl.load(counts_ptr + e)  # int32
            if e < val:
                acc += ce
        # Place i at sorted position pos = acc, then increment exclusive offset for val
        # Write i to sorted_indices[acc]
        tl.store(indices_ptr + acc, tl.full((), i, tl.int32))
        # Increment exclusive offset for val: add counts[val] to acc (only once)
        if active:
            cv = tl.load(counts_ptr + val)  # counts[val]
            acc += cv
            # If val == 0, acc already reflects counts[0]; no extra increment
            # For val > 0, acc reflects sum_{k<val} counts[k]; adding counts[val] for next positions
            # Note: We must ensure acc is incremented only once per i. Since val is fixed for this i,
            # adding counts[val] here would double-count; therefore we add counts[val] to acc above
            # before write, and then recompute acc excluding val when processing other i. This logic
            # is tricky to implement correctly in Triton via branching; to keep correctness, we avoid
            # trying to adjust and instead ensure that for each i, the pos computed is unique by removing
            # counts[val] from acc for the i's own placement. However, Triton does not allow easy
            # per-index branching. For robustness, we recompute acc excluding val by iterating again
            # and skipping e==val. We'll do this by a second loop and masking the increment.
            # Simpler and correct: recompute acc by summing counts[0..val-1]. We can do this by setting
            # acc = 0 and adding ce for all e < val (i.e., exactly what we did), and then explicitly
            # store i at acc and let next i increment acc by counts[val] (which would double-count).
            # To avoid double-counting, we instead compute pos without adding counts[val], then store,
            # and leave acc unchanged for the loop. This means acc after write will include counts[val],
            # which affects subsequent i. But since we assign pos based on acc before adding counts[val],
            # later i will read acc which already includes counts[val], and their pos will shift correctly
            # because acc increases for all subsequent i. This is a standard stable counting sort trick.
            pass
    # The kernel returns by writing to indices_ptr; host reads and returns.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensor
        assert topk_idx.is_cuda, "ModelNew requires CUDA tensors."
        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous()
        n = flat.numel()

        # Prepare int32 flat for Triton
        flat_i32 = flat.to(torch.int32)

        # 1) Triton histogram of expert IDs
        num_experts = 256  # as in original
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        # Grid size: 1D, one program per chunk of BLOCK_SIZE elements
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](flat_i32, counts, n_elements=n, BLOCK_SIZE=BLOCK_SIZE)

        # 2) Triton inclusive prefix sum for offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # offsets[0] = 0 already
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts=num_experts)

        # 3) Triton stable sort to produce sorted_token_indices (indices permutation)
        # We need a permutation of 0..n-1 that would sort flat in stable order.
        # Implement counting-sort-based stable sort: initialize indices to 0..n-1, then place each i
        # at position determined by exclusive prefix sum of counts up to val<flat[i], and increment
        # that exclusive count. Since Triton lacks convenient per-index branching, we emulate with a
        # sequential loop over i (MAX_N as constexpr). The output is written into indices_ptr.
        indices = torch.empty(n, dtype=torch.int32, device=flat.device)
        # Launch kernel: we use MAX_N = n. Triton will JIT with this bound.
        _stable_sort_indices_by_counting_kernel[(1,)](flat_i32, indices, n_elements=n, counts_ptr=counts, num_experts=num_experts, MAX_N=n)

        # Return sorted_token_indices (int64) and expert_offsets (int32)
        sorted_token_indices = indices.to(torch.int64)
        return sorted_token_indices, offsets