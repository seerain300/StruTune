import torch
import triton
import triton.language as tl


@triton.jit
def _global_stable_counting_sort(flat_ptr, out_idx_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Stable global counting sort over the flattened 1D tensor 'flat_ptr' of length N.
    Values are assumed to be in [0, NUM_CLASSES-1] (here NUM_CLASSES=256).
    Writes sorted_token_indices (the permutation) into 'out_idx_ptr'.
    Uses 'counts_ptr' (length NUM_CLASSES) for reservation with atomic adds.
    Each thread handles one element i to maintain stability via tie-breaker offset=i%NUM_CLASSES.
    """
    i = tl.program_id(0)
    if i < N:
        val = tl.load(flat_ptr + i)
        # Read current count for this value
        cnt = tl.load(counts_ptr + val)
        # Stable position: tie-breaker offset ensures equal values keep original order
        offset = i % NUM_CLASSES
        pos = cnt + offset
        tl.store(out_idx_ptr + i, pos)
        # Reserve next slot for next occurrence of this value
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Histogram of flat values into 'counts_ptr' (length NUM_CLASSES).
    Each program handles one class id and scans the entire flat vector to count occurrences.
    """
    cls = tl.program_id(0)
    if cls < NUM_CLASSES:
        total = tl.zeros((), dtype=tl.int32)
        # Iterate over flat in chunks of size NUM_CLASSES
        for off in range(0, N, NUM_CLASSES):
            chunk = tl.arange(0, NUM_CLASSES)
            idx = off + chunk
            mask = idx < N
            vals = tl.load(flat_ptr + idx, mask=mask, other=0)
            # Count occurrences of this class in the chunk
            total += tl.sum((vals == cls).to(tl.int32), axis=0)
        tl.store(counts_ptr + cls, total)


@triton.jit
def _inclusive_scan_kernel(inp_ptr, out_ptr, M: tl.constexpr):
    """
    Inclusive prefix sum over input array of length M, stores result in out_ptr.
    Implements simple sequential scan per element (M is small, e.g., 256).
    """
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, M):
        v = tl.load(inp_ptr + i)
        acc += v
        tl.store(out_ptr + i, acc)


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernel to compute the global stable permutation indices for sorting 'flat'.
    Returns int32 tensor of shape (N,).
    """
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    grid = (N,)
    _global_stable_counting_sort[grid](flat, out_idx, counts, N, NUM_CLASSES=256)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert offsets via Triton histogram + Triton inclusive scan (no torch ops).
    Returns tensor of shape (num_experts + 1,) int32.
    """
    assert flat.is_cuda, "Triton kernels require CUDA tensors"
    N = flat.numel()
    # Triton histogram across the flat tensor
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    grid_hist = (num_experts,)
    _hist_kernel[grid_hist](flat, counts, N, NUM_CLASSES=num_experts)
    # Triton inclusive prefix sum (small vector, num_experts=256)
    inclusive = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    _inclusive_scan_kernel[(num_experts,)](counts, inclusive, M=num_experts)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[0] = 0
    offsets[1:] = inclusive
    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Computes sorted_token_indices = global stable argsort of flattened topk_idx.
        - Computes expert_offsets from original topk_idx via Triton histogram + Triton inclusive scan.
        Returns (sorted_token_indices, expert_offsets).
        """
        # Ensure CUDA tensors for Triton kernels
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda", non_blocking=True)
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        flat = topk_idx.reshape(-1)
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)

        # Sorted token indices via Triton
        sorted_token_indices = _launch_global_sort(flat)  # int32, shape (N,)

        # Expert offsets via Triton histogram + Triton inclusive scan
        expert_offsets = _compute_expert_offsets(flat, 256)  # int32, shape (257,)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
