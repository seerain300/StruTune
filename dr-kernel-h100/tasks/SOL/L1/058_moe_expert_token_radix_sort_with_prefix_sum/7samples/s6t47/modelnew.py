import torch
import triton
import triton.language as tl


# Kernel: compute per-class histogram counts for values in flat (int32)
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # One program runs; it loops over all elements and atomically increments counts
    for i in range(0, N):
        v = tl.load(flat_ptr + i)
        # v is int32; class id is v in [0, NUM_CLASSES-1]
        # For safety, clamp to NUM_CLASSES-1 (though inputs are valid in [0, 255]).
        c = v % NUM_CLASSES
        # Atomic add to counts[c]
        tl.atomic_add(counts_ptr + c, 1)


# Kernel: compute inclusive prefix sums of counts -> scan[0..NUM_CLASSES], length NUM_CLASSES+1
@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    # Single program computes the inclusive scan sequentially
    # Initialize scan[0] = 0
    tl.store(scan_ptr + 0, 0)
    total = 0
    for i in range(0, NUM_CLASSES):
        total += tl.load(counts_ptr + i)
        tl.store(scan_ptr + i + 1, total)


# Triton launcher for histogram
def _launch_histogram(flat: torch.Tensor) -> torch.Tensor:
    N = flat.numel()
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Launch one program
    _hist_kernel[(1,)](flat, counts, N, 256)
    return counts


# Helper to compute inclusive scan of a counts tensor (int32), returns int32 of length NUM_CLASSES+1
def _compute_inclusive_scan(counts: torch.Tensor) -> torch.Tensor:
    NUM_CLASSES = 256
    scan = torch.empty(NUM_CLASSES + 1, dtype=torch.int32, device=counts.device)
    _inclusive_scan_kernel[(1,)](counts, scan, NUM_CLASSES)
    return scan


# Triton kernel: global stable argsort of flat (int32) producing permutation indices (int64)
@triton.jit
def _global_argsort_stable_kernel(flat_ptr, out_idx_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # out_idx_ptr is int64, length N
    for c in range(0, NUM_CLASSES):  # NUM_CLASSES=256
        # start is at scan[c]; since we write one element per i, we must read current start before increment.
        # Triton allows scalar loads/stores; we iterate i, compute current start each time.
        # We need the current start at the moment we process i; we achieve this by loading start before update.
        # Initialize start by loading; we'll load it inside the i loop.
        pass
    # Implement the two-phase stable insertion per class c: loop i from 0..N-1
    for c in range(0, NUM_CLASSES):
        # Determine start position for this class via scan
        # We cannot vectorize this; do it sequentially. Triton supports loops; this is acceptable for given sizes.
        start = tl.load(scan_ptr + c)  # scalar load
        # Iterate i; load value and write i to out_idx[start] if value==c; increment start
        for i in range(0, N):
            v = tl.load(flat_ptr + i)  # int32
            # Check if equal to c
            is_equal = (v == c)
            # Compute address and store i as int64
            # Note: Triton requires vectorized pointers; here we use scalar pointer arithmetic.
            # We'll store scalar i converted to int64 at out_idx[start].
            # We cannot form a vector of addresses easily; rely on scalar store.
            # If is_equal is True, store i
            if is_equal:
                # start is scalar; tl.store with scalar pointer
                tl.store(out_idx_ptr + start, tl.cast(i, tl.int64))
                start += 1  # advance start for next equal element
        # After processing all i, start == scan[c+1]; no need to pad since we wrote exactly count(c) elements.


# Triton launcher for global stable argsort
def _launch_global_argsort(flat: torch.Tensor) -> torch.Tensor:
    N = flat.numel()
    # Output permutation indices as int64
    out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
    # First compute scan for all classes
    counts = _launch_histogram(flat)  # int32 counts length 256
    scan = _compute_inclusive_scan(counts)  # int32 length 257
    # Launch argsort kernel
    _global_argsort_stable_kernel[(1,)](flat, out_idx, scan, N, 256)
    return out_idx


# Triton kernel to compute expert_offsets via histogram and inclusive scan
@triton.jit
def _hist_and_offsets_kernel(flat_ptr, offsets_ptr, N, NUM_CLASSES: tl.constexpr):
    # This kernel is not used directly; the two-step approach (hist + scan) is preferred for clarity and robustness.
    # Keeping this here for completeness, but we will call _launch_histogram and _compute_inclusive_scan separately.
    pass


def _compute_inclusive_scan_from_flat(flat: torch.Tensor) -> torch.Tensor:
    # Same as before
    counts = _launch_histogram(flat)
    scan = _compute_inclusive_scan(counts)
    return scan


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch_size, seq_len, num_experts_per_tok)
        assert topk_idx.dim() == 3, "topk_idx must be 3D: (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D and make contiguous; keep int32 for flat
        flat = topk_idx.reshape(-1).contiguous()
        # sorted_token_indices via Triton global stable argsort (int64, length N)
        sorted_token_indices = _launch_global_argsort(flat)  # int64, shape (N,)
        # expert_offsets via histogram + inclusive scan (int32, length num_experts+1 = 257)
        expert_offsets = _compute_inclusive_scan(_launch_histogram(flat))  # int32, shape (257,)
        # Return exactly as original: sorted_token_indices (N,), expert_offsets (num_experts+1,)
        return sorted_token_indices, expert_offsets[1:]  # return length 256