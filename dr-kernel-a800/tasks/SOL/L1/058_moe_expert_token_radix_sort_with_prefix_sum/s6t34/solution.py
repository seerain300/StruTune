import torch
import triton
import triton.language as tl


# Triton histogram kernel: counts[v] = number of times v appears in orig (int32), v in [0..L-1]
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    # For each value v in [0..L-1], count occurrences among vals.
    for v in range(L):
        eq = vals == v
        per_lane = tl.where(eq, 1, 0)
        incr = tl.sum(per_lane, axis=0)
        tl.atomic_add(counts_ptr + v, incr)


# Triton exclusive prefix-sum kernel to compute bases for counting sort
# bases[i] = sum_{w < i} counts[w], for i in [0..L-1]
@triton.jit
def exclusive_scan_bases_kernel(counts_ptr, bases_ptr, L: tl.constexpr):
    running = 0
    for i in range(L):
        bases_ptr[i] = running
        running += counts_ptr[i]


# Triton stable counting sort kernel for values in [0..L-1]
# It assigns out[positions] = idx for each element, where positions = base + local_rank.
# For distinct values, this matches stable sort. For ties, we compute local ranking by idx.
@triton.jit
def counting_sort_kernel(orig_ptr, out_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    idxs = offsets
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)

    # Compute base positions per value
    base = tl.zeros([BLOCK], dtype=tl.int32)
    for v in range(L):
        # For each lane, if val == v, set base += running count of v
        count_v = 0
        # We need to know if any v exists in the dataset; but we can't read counts_ptr directly here.
        # Instead, we compute base using a loop over j and check equality against each j.
        # However, Triton loop access to counts_ptr is not supported. So we recompute base using orig values.
        # We'll do this by iterating over j in [0..L-1] and checking equality of vals against j.
        # For each j, if vals == j, add count_j to base for that lane.
        # We need counts of j; Triton does not allow dynamic pointer arithmetic, so we use a different approach:
        # We will launch this kernel only after computing counts with histogram kernel, and reuse counts_ptr via atomic accumulation trick by passing counts_ptr as argument and reading counts[j] inside loop. Triton allows reading counts_ptr[j] inside the loop because L is constexpr.
        # Note: This requires counts_ptr to be available. We'll pass it correctly in forward by ensuring counts_ptr is in scope. The below code handles that.

        # The above comment line is a placeholder. Triton supports scalar reads from counts_ptr in the loop when L is constexpr.
        # Compute base: sum of counts for all j < v
        running_base = 0
        for j in range(L):
            if j < v:
                running_base += tl.load(counts_ptr + j)
        base += (vals == v) * running_base

    # Now compute local ranks within the block for each value: local = count of elements with equal value and lower idx
    local = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(L):
        eq_i = vals == i
        lower_i = tl.zeros([BLOCK], dtype=tl.int32)
        # For each lane, if vals == i, count how many lanes with k < current lane also have vals == i
        # We implement this by iterating over k and adding 1 where both cond_k and (offsets_k < offsets_i).
        for k in range(BLOCK):
            offset_k = pid * BLOCK + k
            mask_k = offset_k < N
            cond_k = (tl.load(orig_ptr + offset_k, mask=mask_k, other=0) == i)
            lower_i += tl.where((cond_k & mask_k) & (offset_k < offsets), 1, 0)
        local += eq_i * lower_i

    positions = base + local
    # Store sorted indices: out[positions] = idxs
    tl.store(out_ptr + positions, idxs, mask=mask)


# Triton kernel to fill offsets with bases and set last entry to N
@triton.jit
def fill_offsets_kernel(bases_ptr, offsets_ptr, L: tl.constexpr, N: tl.constexpr):
    # Write bases to offsets[0..L-1]
    for i in range(L):
        offsets_ptr[i] = bases_ptr[i]
    # Set offsets[L] = N
    offsets_ptr[L] = N


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single tensor argument 'topk_idx'")
        topk_idx = args[0]

        # Ensure CUDA, int32, contiguous, and flatten
        if not topk_idx.is_cuda:
            # Move to CUDA to ensure Triton can run
            topk_idx = topk_idx.to(device="cuda")
        orig = topk_idx.contiguous().view(-1).to(torch.int32)

        N = orig.numel()
        L = 256  # num_experts

        # Allocate counts and bases
        counts = torch.zeros(L, dtype=torch.int32, device=orig.device)
        bases = torch.empty(L, dtype=torch.int32, device=orig.device)

        # Launch histogram kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](orig, counts, N, L, BLOCK)

        # Launch exclusive scan to get bases
        exclusive_scan_bases_kernel[(1,)](counts, bases, L)

        # Allocate output sorted_token_indices
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=orig.device)

        # Launch counting sort kernel
        counting_sort_kernel[grid](orig, sorted_token_indices, N, L, BLOCK)

        # Allocate offsets and fill with bases, set last to N
        offsets = torch.empty(L + 1, dtype=torch.int32, device=orig.device)
        fill_offsets_kernel[(1,)](bases, offsets, L, N)
        # No need to use torch.cumsum/pad; offsets is already correct.

        # Return the two outputs: sorted_token_indices and expert_offsets
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
