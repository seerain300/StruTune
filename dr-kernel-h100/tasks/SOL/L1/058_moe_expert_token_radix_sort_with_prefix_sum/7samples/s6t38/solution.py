import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # Compute histogram of values in flat (int32) into counts (int32), one program loops over N.
    # Values are expected to be in [0, NUM_CLASSES-1].
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        # Atomic add 1 to the bin corresponding to val
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    # Compute inclusive prefix sums of counts into scan.
    # scan[0] = 0; scan[i] = scan[i-1] + counts[i-1] for i=1..NUM_CLASSES
    # We use a single program to perform the scan serially; NUM_CLASSES is small (256).
    # Initialize scan[0] = 0
    tl.store(scan_ptr + 0, 0)
    # Compute prefix sums sequentially
    for i in range(1, NUM_CLASSES + 1):
        prev = tl.load(scan_ptr + (i - 1))
        cur = tl.load(counts_ptr + (i - 1))
        tl.store(scan_ptr + i, prev + cur)


@triton.jit
def _global_argsort_stable_kernel(flat_ptr, out_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # Fill out_ptr with the permutation indices that would sort flat ascending (stable).
    # We iterate over classes and then iterate over tokens in increasing order to maintain stability.
    for class_id in range(0, NUM_CLASSES):
        start = tl.load(scan_ptr + class_id)  # inclusive start for this class
        next_pos = start  # exclusive end for this class
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            if val == class_id:
                tl.store(out_ptr + i, next_pos)
                next_pos += 1
        # No need to set scan_ptr[class_id+1] here; the caller computes it.


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Compute sorted_token_indices using Triton kernels:
    - Histogram of values
    - Inclusive scan of counts
    - Stable fill of output permutation
    Returns out_idx of shape (N,), dtype int32.
    """
    N = flat.numel()
    NUM_CLASSES = 256
    # Ensure flat is int32
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    flat = flat.contiguous()
    # Allocate counts and scan
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int32, device=flat.device)
    scan = torch.empty(NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
    # Launch histogram kernel
    _hist_kernel[(1,)](flat, counts, N, NUM_CLASSES)
    # Launch inclusive scan kernel
    _inclusive_scan_kernel[(1,)](counts, scan, NUM_CLASSES)
    # Allocate output permutation
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    # Launch stable argsort fill
    _global_argsort_stable_kernel[(1,)](flat, out_idx, scan, N, NUM_CLASSES)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor) -> torch.Tensor:
    """
    Compute expert_offsets of shape (num_experts + 1,) = (257,) using Triton:
    - Histogram of values
    - Inclusive scan to get cumulative counts
    Returns scan[1:] which matches original offsets[1:].
    """
    NUM_CLASSES = 256
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    flat = flat.contiguous()
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int32, device=flat.device)
    scan = torch.empty(NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
    _hist_kernel[(1,)](flat, counts, flat.numel(), NUM_CLASSES)
    _inclusive_scan_kernel[(1,)](counts, scan, NUM_CLASSES)
    # Original returns offsets of length (num_experts + 1), set [1:] from scan
    return scan[1:]


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Get inputs as in original helper
        # No need to call get_inputs; we receive topk_idx directly.
        # Ensure 3D input
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous()

        # Compute sorted_token_indices via Triton
        sorted_token_indices = _launch_global_sort(flat)  # shape: (N,), int32

        # Compute expert_offsets via Triton (shape: (num_experts + 1,) = (257,), int32)
        expert_offsets = _compute_expert_offsets(flat)

        # Return with exact shapes/dtypes as original
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
