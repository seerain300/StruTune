import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    # Compute per-class counts from flat via atomic adds.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)  # int32
        # We assume val in [0, NUM_CLASSES-1]; atomic add 1 to counts[val]
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.int32):
    # Compute inclusive scan (prefix sum) of counts into scan:
    # scan[k] = sum_{j=0..k} counts[j]
    total = 0
    for k in range(0, NUM_CLASSES):
        total += tl.load(counts_ptr + k)
        tl.store(scan_ptr + k, total)


@triton.jit
def _global_argsort_fill_kernel(flat_ptr, out_idx_ptr, scan_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    # Fill out_idx with stable global argsort permutation.
    # For each class c: place indices i where flat[i] == c at scan[c], increment scan[c] by 1.
    for c in range(0, NUM_CLASSES):
        ptr = scan_ptr + c  # int32 pointer to current scan slot
        # Loop over all tokens i; Triton allows Python loops, but we keep it simple.
        for i in range(0, N):
            vi = tl.load(flat_ptr + i)
            if vi == c:
                tl.store(out_idx_ptr + i, tl.load(ptr))
                tl.atomic_add(ptr, 1)  # advance to next position for class c


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernels to compute sorted_token_indices via global stable counting sort.
    Returns a 1D tensor of int32 with length N (flattened).
    """
    N = flat.numel()
    # Triton kernels expect int32
    if flat.dtype != torch.int32:
        flat_i32 = flat.to(torch.int32)
    else:
        flat_i32 = flat

    # Allocate output permutation
    out_idx = torch.empty(N, dtype=torch.int32, device=flat_i32.device)

    # NUM_CLASSES is fixed to 256 in this problem
    NUM_CLASSES = 256
    # Count per class
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int32, device=flat_i32.device)
    _hist_kernel[(1,)](flat_i32, counts, N, NUM_CLASSES)

    # Inclusive prefix sums to get scan
    scan = torch.empty(NUM_CLASSES, dtype=torch.int32, device=flat_i32.device)
    _inclusive_scan_kernel[(1,)](counts, scan, NUM_CLASSES)

    # Fill out_idx stably
    _global_argsort_fill_kernel[(1,)](flat_i32, out_idx, scan, N, NUM_CLASSES)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor) -> torch.Tensor:
    """
    Compute expert_offsets using Triton histogram and inclusive scan, returning
    a 1D tensor of length (num_experts + 1) = 257, dtype int32.
    """
    N = flat.numel()
    # We can compute counts directly from flat; values are in [0, 255].
    NUM_CLASSES = 256
    counts = torch.zeros(NUM_CLASSES, dtype=torch.int32, device=flat.device)
    _hist_kernel[(1,)](flat.to(torch.int32), counts, N, NUM_CLASSES)

    scan = torch.empty(NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
    # Initialize scan[0] = 0
    scan[0] = 0
    # Inclusive scan of counts into scan[1:]
    _inclusive_scan_kernel[(1,)](counts, scan[1:], NUM_CLASSES)
    return scan[1:]  # shape: (256 + 1 - 1:) -> (257,)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure 3D input
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D, keep original device/dtype handling
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device

        # 1) Compute sorted_token_indices (global stable argsort) via Triton
        sorted_token_indices = _launch_global_sort(flat)  # shape: (N,), int32

        # 2) Compute expert_offsets via Triton histogram + inclusive scan
        #    Return shape: (num_experts + 1,) = (257,), int32
        expert_offsets = _compute_expert_offsets(flat)

        return sorted_token_indices, expert_offsets