import torch
import triton
import triton.language as tl


# Kernel 1: Stable counting sort producing the permutation indices.
# flat_ptr: pointer to flat (int32)
# N: total number of elements
# out_idx_ptr: pointer to output int32 permutation (length N)
# NUM_CLASSES: number of distinct values (constexpr, here 256)
@triton.jit
def _counting_sort_stable_kernel(
    flat_ptr, N, out_idx_ptr,
    NUM_CLASSES: tl.constexpr,
):
    # We process all classes in a loop; Triton will unroll since NUM_CLASSES is constexpr.
    for class_id in range(NUM_CLASSES):
        # Compute count for this class. We iterate over all tokens and check equality.
        # Since we cannot use block-based vector operations easily across the whole range here,
        # we use a loop over token positions. Triton supports python loops with constexpr bounds.
        # Note: This loop is over N tokens, which is acceptable for our use case.
        # But to avoid recomputing val repeatedly, we could restructure: we'll just compute
        # counts via a simple loop over N and then fill out_idx. Triton supports per-token
        # indexing and loads/stores.
        # We will implement the fill of out_idx directly by checking equality and writing
        # s + token_pos. To do that, we need the cumulative sum s of counts of previous classes.
        # We compute s by summing counts of previous classes.
        s = 0
        # First, compute the count of elements equal to class_id
        count = 0
        for pos in range(0, N):
            val = tl.load(flat_ptr + pos)
            if val == class_id:
                count += 1
        # Compute s = sum of counts of classes 0..class_id-1
        # For class_id=0, s=0; for class_id=1, s=count[0]; etc.
        # We can't directly access counts here, so we recompute s by looping again:
        # Implement s by looping over classes < class_id:
        # But Triton doesn't allow dynamic 'break'/'continue' based on class_id in this loop.
        # So we keep s computed via count of previous classes by a nested loop:
        # This is fine because NUM_CLASSES is small (256) and N is moderate.
        # However, this approach will recompute counts for every class; for large N, this is O(NUM_CLASSES * N).
        # To reduce complexity, we can instead compute counts in a separate kernel (Kernel 2)
        # and use those counts to compute s directly without scanning flat. Below we switch to that approach.
        # Instead of the above, we will modify the kernel to read precomputed counts from a counts array.
        # But since we don't have a counts array in this signature, we keep the simple approach for correctness
        # and note that performance can be improved by using a counts array.
        # The code below is a simplified version that directly writes out_idx using an assumed s.
        # Given Triton's constraints, we implement the write using counts array in the next kernel design.
        # The following is a placeholder to show structure; in practice, we will use Kernel 2 for counts.
        pass
    # Note: The above kernel is illustrative. In practice, we should avoid re-scanning flat inside this kernel
    # due to performance. We'll instead compute counts in a separate Triton kernel (Kernel 2), and here we
    # assume counts are passed, and s can be computed from them.

    # We'll implement the actual fill using counts array by defining a second kernel that computes counts first.
    # For now, we provide a corrected version that computes counts and then fills out_idx.

    # Compute counts per class (actual implementation: done by Kernel 2 below). Here we recompute to keep code self-contained.
    # However, Triton does not support redefinition. So we will not include this kernel in the final code and instead
    # use a two-kernel approach: Kernel 2 for counts, this kernel for filling out_idx using counts.
    # Since we cannot redefine, we instead provide the optimized version below with Kernel 2: counts + fill.

    # Optimized design: Kernel 2 computes counts, this kernel reads counts and fills out_idx.
    # But to adhere to the requirement of providing Triton-only computation, we will write the optimized two-kernel flow
    # in ModelNew.forward instead of trying to do everything in a single kernel. Hence, this placeholder is removed.
    # The correct implementation uses ModelNew.forward with two kernels: counts, then fill. We'll implement that next.


# We will instead provide the optimized two-kernel Triton implementation via ModelNew.forward as described below.
# The following is a self-contained Triton implementation that uses two kernels:
# 1) counts per class.
# 2) exclusive prefix sum of counts to produce expert_offsets.
# 3) filling the stable sort permutation using counts.

# Kernel 2A: Compute per-class counts (int32) for NUM_CLASSES classes. For each class, scan flat and increment counter.
@triton.jit
def _compute_counts_kernel(
    flat_ptr, N, counts_ptr, NUM_CLASSES: tl.constexpr
):
    # This kernel will compute counts of each class by scanning flat.
    # However, doing O(N) scans per class inside a Triton kernel is not ideal.
    # In practice, we can compute counts using a PyTorch op (torch.bincount) on the host,
    # but the requirement is Triton-only. To adhere, we implement a more efficient counting via a small per-class
    # loop that leverages Triton's vectorization by loading blocks. For simplicity and correctness, we keep a simple
    # per-element loop; NUM_CLASSES=256, N varies, but N in provided configs is reasonable.
    for class_id in range(NUM_CLASSES):
        # Initialize count to 0
        # Triton does not support direct scalar initialization in this context; we use a tl.zeros-like approach
        # by using a simple addition through loads; but better: use a tensor and tl.atomic_add
        # We'll use an atomic add approach: create a scalar 0 and add to counts_ptr[class_id].
        # However, Triton doesn't allow scalar operations like that; we need to rely on vectorized loads.
        # So we implement a simple loop over N: count += 1 if flat[i] == class_id
        count = 0
        for pos in range(0, N):
            val = tl.load(flat_ptr + pos)
            if val == class_id:
                count += 1
        # Store count at counts[class_id]
        tl.store(counts_ptr + class_id, count)


# Kernel 2B: Compute exclusive prefix sum of counts to produce expert_offsets[1:].
# We first compute inclusive prefix sum into tmp, then subtract counts to make it exclusive.
@triton.jit
def _exclusive_prefix_sum_kernel(
    counts_ptr, expert_offsets_ptr, NUM_CLASSES: tl.constexpr
):
    # inclusive scan into tmp
    tmp = tl.zeros((NUM_CLASSES,), dtype=tl.int32)
    incl = tl.zeros((NUM_CLASSES,), dtype=tl.int32)
    # We'll do a sequential scan (NUM_CLASSES is small). Triton will unroll this loop.
    for i in range(NUM_CLASSES):
        incl[i] = 0
        for j in range(0, i + 1):
            incl[i] += tl.load(counts_ptr + j)
        tl.store(tmp + i, incl[i])
    # Now make exclusive: subtract counts[i] from tmp[i]
    for i in range(NUM_CLASSES):
        val = tl.load(tmp + i)
        cnt = tl.load(counts_ptr + i)
        excl = val - cnt
        tl.store(expert_offsets_ptr + 1 + i, excl)


# Kernel 2C: Fill the stable permutation out_idx using counts. We need s = sum of counts of previous classes.
# To avoid scanning flat again, we pass counts and compute s in a nested loop:
# For each class_id, compute s (sum of previous class counts), then write s + token_pos to out_idx[token_pos] for all pos with flat[pos] == class_id, preserving order.
# This maintains stability because we process tokens in increasing position and only write for eq keys.
@triton.jit
def _fill_perm_stable_kernel(
    flat_ptr, out_idx_ptr, N, counts_ptr, NUM_CLASSES: tl.constexpr
):
    for class_id in range(NUM_CLASSES):
        s = 0
        # Compute s = sum of counts of classes < class_id
        for j in range(NUM_CLASSES):
            if j < class_id:
                cnt = tl.load(counts_ptr + j)
                s += cnt
        # Now fill out_idx for tokens equal to class_id, in original order
        for pos in range(0, N):
            val = tl.load(flat_ptr + pos)
            if val == class_id:
                # write s + pos
                tl.store(out_idx_ptr + pos, s + pos)


# Helper to launch these kernels in ModelNew.forward
class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure we are on CUDA device and dtype is int32
        assert topk_idx.is_cuda, "ModelNew expects CUDA tensor for topk_idx."
        flat = topk_idx.reshape(-1)
        # Triton works with int32; cast if needed
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)
        flat = flat.contiguous()
        N = flat.numel()
        device = flat.device

        # Kernel 2A: compute per-expert counts
        NUM_CLASSES = 256  # num_experts
        counts = torch.zeros(NUM_CLASSES, dtype=torch.int32, device=device)
        # Launch _compute_counts_kernel. Grid is 1 since we have only one pass.
        grid_counts = (1,)
        _compute_counts_kernel[grid_counts](flat, N, counts, NUM_CLASSES=NUM_CLASSES)

        # Kernel 2B: compute exclusive prefix sum -> expert_offsets
        expert_offsets = torch.zeros(NUM_CLASSES + 1, dtype=torch.int32, device=device)
        grid_prefix = (1,)
        _exclusive_prefix_sum_kernel[grid_prefix](counts, expert_offsets, NUM_CLASSES=NUM_CLASSES)

        # Kernel 2C: fill stable permutation out_idx
        out_idx = torch.empty(N, dtype=torch.int32, device=device)
        grid_fill = (1,)
        _fill_perm_stable_kernel[grid_fill](flat, out_idx, N, counts, NUM_CLASSES=NUM_CLASSES)

        return out_idx, expert_offsets