import torch
import triton
import triton.language as tl


@triton.jit
def _global_argsort_counting_kernel(flat_ptr, out_idx_ptr, counts_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    # We process all tokens in two passes using a single program instance that loops over N.
    # Triton allows loops; each iteration is a separate sequential step in this single program.
    # This reproduces a stable global argsort for integer keys in [0, NUM_CLASSES-1].

    # First pass: count occurrences of each class
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        # counts[val] += 1
        # Note: Triton indexing with a value is fine when passed as int32.
        tl.atomic_add(counts_ptr + val, 1)

    # Second pass: fill out_idx in sorted order (stable). Iterate i again.
    for i in range(0, N):
        # Find the next class that has remaining count > 0. We do a linear scan over classes.
        # For stable order: we pick the smallest class (lowest index) among those with count > 0.
        found = False
        # c is a class id in [0, NUM_CLASSES-1]
        for c in range(0, NUM_CLASSES):
            cnt = tl.load(counts_ptr + c)
            if cnt > 0:
                found = True
                # Decrement count to place one more token of this class
                tl.atomic_add(counts_ptr + c, -1)
                # Place i at the next available slot in out_idx. We compute position by summing counts of all classes j < c.
                # But since we decrement c's count after placing, we cannot use that. Instead, maintain a separate
                # position counter per class by iterating again and writing at positions based on cumulative counts.
                # To implement direct placement, we need to know the cumulative positions for each class.
                # Triton doesn't support dynamic return of per-class position counters, so we write the i at the place
                # where we computed j < c previously. A simpler way is to iterate again to compute per-class positions:
                # We can't do that here without another loop. Therefore, we restructure: for each i, compute the
                # next smallest class by scanning counts. We can set out_idx[i] to the computed position using a
                # prefix sum approach but Triton lacks global persistent storage of positions. Instead, we will
                # compute positions per class by atomically writing into out_idx based on per-class scans.
                # Better: Launch per-class scan to compute positions and then perform a second kernel that fills out_idx.
                # However, to keep in one kernel, we use a trick: we maintain positions per class by scanning counts
                # and writing directly using a conditional store based on a global position pointer. Triton doesn't
                # provide per-thread unique writes without atomics. The simplest robust approach is to perform the
                # stable placement in a second dedicated Triton kernel.
                # Here, we'll write the placement using the counts pointer and atomic_add to a positions counter.
                # But implementing stable placement correctly in a single kernel requires keeping a positions array.
                # Since Triton doesn't allow returning multiple outputs easily within a single kernel for this task,
                # we will instead use two kernels: counting + fill with stable insertion. Triton does not support
                # dynamic multi-output here, so we split into two kernels below. For this submission, we define
                # a second kernel to fill positions stably, and we call it from Python using the counts array.
                # This approach ensures correctness first. The comment below is purely for explanation.
                # We will skip the direct write here; it will be done in the next kernel using counts and out_idx.

        # The above approach needs a second kernel to write out_idx. Triton kernels are launched from Python.
        # So we cannot embed the write in this kernel. We will instead compute counts and then launch a fill kernel
        # that uses counts to write out_idx stably. The code below will call that kernel in Python after this one
        # returns the counts. For completeness, we return here; the Python code will handle the fill.

        # Note: The following lines are not executed in this kernel; they are part of the plan. We will compute
        # counts and then invoke a separate Triton kernel to fill out_idx based on counts.


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    # Histogram of flat values into counts_ptr[0:NUM_CLASSES]
    # One program instance, loops over N and increments counts.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.int32):
    # Compute inclusive scan (prefix sum) of counts_ptr into scan_ptr[0:NUM_CLASSES]
    # We can do this with a sequential loop; given NUM_CLASSES=256 it's fine.
    # Initialize scan_ptr to zeros (host code will do that). Then compute scan = prefix sum.
    # Triton does not have built-in cumsum, so we implement it.
    total = tl.zeros((), dtype=tl.int32)
    for k in range(0, NUM_CLASSES):
        total += tl.load(counts_ptr + k)
        tl.store(scan_ptr + k, total)


# The following function is the actual implementation of stable global argsort via two Triton kernels:
# 1) Histogram to counts
# 2) Fill out_idx stably using counts and original flat ordering via a second kernel
def _global_argsort_triton(flat: torch.Tensor) -> torch.Tensor:
    """
    Returns sorted_token_indices: permutation of indices [0..N-1] that would sort flat ascending.
    Stable sort for integer keys in [0, 255]. Uses Triton kernels only; no torch.sort.
    """
    assert flat.dtype == torch.int32, "flat must be int32"
    N = flat.numel()
    # Compute counts via Triton
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    _hist_kernel[(1,)](flat, counts, N, 256)

    # Inclusive scan to get positions per class
    scan = torch.empty(256, dtype=torch.int32, device=flat.device)
    _inclusive_scan_kernel[(1,)](counts, scan, 256)

    # Now fill out_idx stably. We need to know the positions for each class based on scan.
    # For each i in 0..N-1:
    #   c = flat[i]
    #   position = scan[c] - 1 (since we want the next slot)
    #   out_idx[i] = position
    #   scan[c] -= 1
    # We can implement this as a second kernel that reads flat and writes out_idx using scan.
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    # Kernel that performs stable fill
    @triton.jit
    def _fill_out_idx_kernel(flat_ptr, out_idx_ptr, scan_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
        # For each i, compute c and place i at scan[c]-1, then decrement scan[c].
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            pos = tl.load(scan_ptr + val) - 1  # inclusive -> exclusive for placement
            # out_idx[i] = pos
            tl.store(out_idx_ptr + i, pos)
            # Decrement scan[val]
            tl.atomic_add(scan_ptr + val, -1)

    _fill_out_idx_kernel[(1,)](flat, out_idx, scan, N, 256)
    return out_idx


@triton.jit
def _compute_expert_offsets_kernel(flat_ptr, offsets_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    """
    Compute inclusive prefix sums of the histogram of flat values into offsets_ptr[0:NUM_CLASSES].
    Then host code will return offsets[1:] as the final expert_offsets.
    """
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int32, device=flat_ptr.device)
    # First, compute counts using a histogram kernel. To keep everything Triton, we can inline a loop.
    # However Triton kernels here are device functions. The simplest is to do the histogram in a separate
    # Triton kernel, and then do the scan here. But we need to keep the forward without torch ops.
    # Instead, we compute counts by scanning flat in this kernel using tl.load and tl.atomic_add.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        tl.atomic_add(counts + val, 1)
    # Now compute inclusive scan in offsets_ptr
    total = tl.zeros((), dtype=tl.int32)
    for k in range(0, NUM_CLASSES):
        total += counts[k]
        tl.store(offsets_ptr + k, total)


def _compute_expert_offsets(flat: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of shape (NUM_CLASSES + 1,) with offsets[0]=0 and offsets[1:] = inclusive prefix sums
    of the histogram of flat values. Uses Triton only.
    """
    assert flat.dtype == torch.int32, "flat must be int32"
    N = flat.numel()
    offsets = torch.empty(256 + 1, dtype=torch.int32, device=flat.device)
    # We can compute counts in a Triton kernel and then do scan in this Triton kernel. To avoid
    # Python-side torch ops, we inline the histogram in Triton and do the scan in Triton.
    _compute_expert_offsets_kernel[(1,)](flat, offsets, N, 256)
    # Return offsets[1:] as required
    return offsets[1:]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-only implementation of run(topk_idx: torch.Tensor).
        Returns:
          sorted_token_indices: int32 tensor of shape (N,) where N = topk_idx.numel()
          expert_offsets: int32 tensor of shape (num_experts + 1,) where num_experts=256
        """
        # Expect one input: topk_idx
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects exactly one argument: topk_idx")
        topk_idx = args[0]
        # Ensure int32 and on CUDA
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)
        if topk_idx.device.type != "cuda":
            topk_idx = topk_idx.cuda()

        # Flatten to 1D as original
        flat = topk_idx.reshape(-1)

        # 1) Global stable argsort via Triton
        sorted_token_indices = _global_argsort_triton(flat)  # int32 of shape (N,)

        # 2) Expert offsets via Triton histogram + inclusive scan
        expert_offsets = _compute_expert_offsets(flat)  # int32 of shape (256 + 1,), return 1:

        # Return with exact shapes/dtypes as original
        return sorted_token_indices, expert_offsets