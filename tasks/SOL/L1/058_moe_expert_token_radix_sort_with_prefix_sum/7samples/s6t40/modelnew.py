import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Compute per-class counts of flat values (int32).
    flat_ptr: *int32, 1D, length N
    counts_ptr: *int32, 1D, length NUM_CLASSES
    N: int, total elements
    NUM_CLASSES: compile-time int (256 here)
    """
    # Each program handles a block of elements
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N

    # Load values
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Ensure values are within valid range (already assumed in [0, NUM_CLASSES-1])
    # For each value, atomic add to the corresponding counts entry
    for i in range(NUM_CLASSES):
        eq = vals == i
        idxs = offs & (~eq)  # create indices where eq is True
        # tl.atomic_add expects a pointer and a scalar; we sum eq mask and add
        tl.atomic_add(counts_ptr + i, tl.sum(idx for idx in tl.where(eq, 1, 0)))


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    """
    Inclusive scan of counts[0..NUM_CLASSES-1] -> scan[0..NUM_CLASSES], with scan[0]=0.
    counts_ptr: *int32, length NUM_CLASSES
    scan_ptr: *int32, length (NUM_CLASSES + 1)
    """
    # Single-program scan to build prefix sums
    s = 0
    for i in range(NUM_CLASSES + 1):
        if i == 0:
            val = 0
        else:
            val = s + tl.load(counts_ptr + (i - 1))
            s = val
        tl.store(scan_ptr + i, val)


@triton.jit
def _global_argsort_fill_kernel(flat_ptr, out_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Fill out_ptr[0..N-1] with the original index i at position start for each class c,
    where start is the exclusive scan of counts for that class. Stable sort.
    flat_ptr: *int32, 1D, length N
    out_ptr: *int32, 1D, length N
    scan_ptr: *int32, 1D, length (NUM_CLASSES + 1)
    N: int
    NUM_CLASSES: compile-time int
    """
    # Single-program fill loop over classes, then original indices
    for c in range(NUM_CLASSES):
        start = tl.load(scan_ptr + c)  # exclusive start index for class c
        end = tl.load(scan_ptr + (c + 1))  # exclusive end index
        # Iterate original indices to place them stably
        for i in range(N):
            vi = tl.load(flat_ptr + i)
            if vi == c:
                tl.store(out_ptr + i, start)
                start += 1


def _compute_expert_offsets(flat: torch.Tensor) -> torch.Tensor:
    """
    Compute expert_offsets of shape (num_experts + 1,) via Triton:
    offsets[1:] = inclusive cumulative counts of tokens per expert in flat.
    """
    assert flat.dtype == torch.int32
    assert flat.is_cuda
    N = flat.numel()
    num_experts = 256
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

    # Launch histogram kernel
    grid = (triton.cdiv(N, 1024),)
    _hist_kernel[grid](flat, counts, N, NUM_CLASSES=num_experts)

    # Compute inclusive scan (exclusive scan result will be scan[1:])
    scan = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    _inclusive_scan_kernel[(1,)](counts, scan, NUM_CLASSES=num_experts)
    # offsets[0] = 0 by construction; [1:] equals scan[1:]
    offsets = scan
    return offsets


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernels to compute sorted_token_indices (global stable argsort).
    Returns a 1D tensor of length N, dtype int32.
    """
    assert flat.dtype == torch.int32 and flat.is_cuda
    N = flat.numel()
    num_experts = 256
    # Scan buffer (exclusive scan)
    scan = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

    # 1) Histogram to get counts
    grid = (triton.cdiv(N, 1024),)
    _hist_kernel[grid](flat, scan[1:], N, NUM_CLASSES=num_experts)

    # 2) Exclusive scan: compute inclusive scan to get start indices per class
    _inclusive_scan_kernel[(1,)](scan[1:], scan, NUM_CLASSES=num_experts)

    # 3) Fill output permutation
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    _global_argsort_fill_kernel[(1,)](flat, out_idx, scan, N, NUM_CLASSES=num_experts)
    return out_idx


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is 3D as per the original helper: (batch, seq_len, num_experts_per_tok)
        assert topk_idx.dim() == 3, "topk_idx must be a 3D tensor (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D and ensure int32, CUDA
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        assert flat.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # 1) sorted_token_indices: global stable argsort
        sorted_token_indices = _launch_global_sort(flat)  # shape: (N,), int32

        # 2) expert_offsets: shape (num_experts + 1,) = (257,), int32
        expert_offsets = _compute_expert_offsets(flat)  # length 257

        # Return results on the same device as input
        return sorted_token_indices, expert_offsets