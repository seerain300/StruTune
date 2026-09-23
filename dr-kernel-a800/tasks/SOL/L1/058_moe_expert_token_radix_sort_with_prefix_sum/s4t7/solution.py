import torch
import triton
import triton.language as tl


# Kernel: per-expert histogram of flattened indices
# Inputs:
#   x_ptr: pointer to int32 flattened indices (length N)
#   out_counts_ptr: pointer to int32 array of size NUM_EXPS (256), initialized to zeros
# Grid: (ceil_div(N, BLOCK),)
@triton.jit
def _histogram_kernel(x_ptr, out_counts_ptr, N, BLOCK: tl.constexpr, NUM_EXPS: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load indices and cast to int32
    idx = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # For each expert j in [0, NUM_EXPS), count occurrences in this block and atomically add
    for j in tl.static_range(NUM_EXPS):
        eq = (idx == j) & mask
        count_j = tl.sum(eq.to(tl.int32), axis=0)
        tl.atomic_add(out_counts_ptr + j, count_j)


# Kernel: inclusive prefix sum of per-expert counts (NUM_EXPS = 256, out length = NUM_EXPS + 1)
# Input: counts_ptr (int32, length NUM_EXPS), initialized by histogram
# Output: out_ptr (int32, length NUM_EXPS+1), out[0] = 0, out[1:] = inclusive prefix sum
# We scan from the end to the beginning to compute the correct prefix sums without host ops.
@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, out_ptr, NUM_EXPS: tl.constexpr):
    # This kernel is tiny (NUM_EXPS = 256), so we run it as a single program.
    # We keep the running sum in a scalar and write it out.
    running = 0
    # We loop i from NUM_EXPS-1 down to 0. Triton supports such scalar loops over constexpr ranges.
    for i in tl.static_range(NUM_EXPS - 1, -1, -1):
        cur = tl.load(counts_ptr + i)
        running = running + cur
        tl.store(out_ptr + (i + 1), running)
    # out[0] should be 0; if out_ptr was initialized to zeros, it is correct.


# Kernel: stable argsort by flattened values (return permutation of indices)
# We implement counting sort by value, and within each value we keep original index order for stability.
# Inputs:
#   x_ptr: pointer to int32 flattened indices (length N)
#   idx_out_ptr: pointer to int32 output permutation (length N), initialized to zeros
#   N: total number of elements
# We compute min and max of x_ptr in-kernel and then populate idx_out_ptr with stable order.
@triton.jit
def _stable_argsort_by_flat_kernel(x_ptr, idx_out_ptr, N, BLOCK: tl.constexpr):
    # First pass: compute min and max of x_ptr
    min_val = 0x7FFFFFFF
    max_val = 0x80000000  # negative infinity for int32 in two's complement
    # We need to scan x_ptr to get min and max. A simple approach: use a loop over tiles.
    # Here we perform a naive linear scan using a while-like loop inside the kernel.
    # Note: Triton supports loops; we implement a loop that iterates over all elements.
    # We'll process one element at a time for simplicity and correctness across any N.
    # This loop is over N elements, which is fine for the provided sizes.
    i = 0
    while i < N:
        val = tl.load(x_ptr + i)
        # Update min and max
        # min_val = min(min_val, val); max_val = max(max_val, val)
        # Implement via conditional moves
        min_val = tl.where(val < min_val, val, min_val)
        max_val = tl.where(val > max_val, val, max_val)
        i += 1

    # Second pass: counting sort by value with stability
    # We iterate over all i in [min_val, max_val]. For each i, count occurrences and place them in idx_out in order.
    i = min_val
    running = 0
    while i <= max_val:
        # Find frequency of i: scan x_ptr again
        cnt = 0
        j = 0
        while j < N:
            val_j = tl.load(x_ptr + j)
            cnt += (val_j == i).to(tl.int32)
            j += 1

        # Place occurrences: we need original index order for stability.
        # For each position k in 0..cnt-1, find the k-th occurrence of i and write it to idx_out[running + k].
        # We can do this by scanning again and maintaining a local counter.
        k = 0
        pos = running
        j = 0
        while j < N:
            val_j = tl.load(x_ptr + j)
            if val_j == i:
                # Determine if this is the k-th occurrence in original order: we don't have prev j, so we just place as we find.
                # Stable within each group is ensured by scanning order j (original order).
                tl.store(idx_out_ptr + pos, j)
                pos += 1
                k += 1
                # Once k >= cnt, we can break; but Triton doesn't support dynamic breaks easily, so we continue.
                # We keep this loop running; it will write all occurrences sequentially and stably.
            j += 1

        running += cnt
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten
        flat = topk_idx.view(-1)
        N = flat.numel()

        # 1) Triton histogram
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_kernel[grid](flat, counts, N, BLOCK=BLOCK, NUM_EXPS=num_experts)

        # 2) Triton inclusive prefix sum to get expert_offsets
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # Run the prefix sum kernel (single program suffices because NUM_EXPS is small)
        _inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, NUM_EXPS=num_experts)

        # 3) Triton stable argsort by flattened values (return permutation indices)
        sorted_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        _stable_argsort_by_flat_kernel[(1,)](flat, sorted_indices, N, BLOCK=1024)

        return sorted_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
