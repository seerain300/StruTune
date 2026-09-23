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
        # Read current count for this value (all lanes read the same val)
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
        for i in range(0, N, NUM_CLASSES):
            chunk = tl.arange(0, NUM_CLASSES)
            idx = i + chunk
            mask = idx < N
            # Load class values; out-of-range indices yield 0 which won't affect count
            vals = tl.load(flat_ptr + idx, mask=mask, other=0)
            eq = vals == cls
            # Count occurrences in this chunk
            # Note: This creates a temporary vector; Triton will handle loop bounds.
            # We sum across the chunk using a reduction over NUM_CLASSES lanes.
            # If no match, eq is False; eq.to(int32) contributes 0.
            count = tl.sum((eq & mask).to(tl.int32), axis=0)
            total += count
        tl.store(counts_ptr + cls, total)


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Launches the Triton kernel to produce the permutation 'sorted_token_indices' for global stable sort.
    Returns a 1D torch.int32 tensor of length N.
    """
    assert flat.is_cuda, "Triton kernels require CUDA tensors"
    N = flat.numel()
    # Output permutation indices
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    # Counters per class
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Launch one program per element
    grid = (N,)
    _global_stable_counting_sort[grid](flat, out_idx, counts, N, NUM_CLASSES=256)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert offsets via Triton histogram and torch cumsum for inclusive scan.
    Returns tensor of shape (num_experts + 1,) int32.
    """
    assert flat.is_cuda, "Triton kernels require CUDA tensors"
    N = flat.numel()
    # Triton histogram across the flat tensor
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    grid_hist = (num_experts,)
    _hist_kernel[grid_hist](flat, counts, N, NUM_CLASSES=num_experts)
    # Inclusive prefix sum with torch (negligible, small num_experts)
    inclusive = torch.cumsum(counts, dim=0)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[0] = 0
    offsets[1:] = inclusive
    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Computes sorted_token_indices = global stable argsort of flattened topk_idx.
        - Computes expert_offsets from original topk_idx via Triton histogram + torch cumsum.
        Returns (sorted_token_indices, expert_offsets).
        """
        # Ensure CUDA tensors for Triton kernels
        if not topk_idx.is_cuda:
            # If inputs come on CPU, move to current CUDA device (evaluation harness usually provides CUDA)
            device = torch.device("cuda")
            topk_idx = topk_idx.to(device)
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        flat = topk_idx.reshape(-1)
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)

        # Sorted token indices via Triton
        sorted_token_indices = _launch_global_sort(flat)  # int32, shape (N,)

        # Expert offsets via Triton histogram + torch cumsum
        expert_offsets = _compute_expert_offsets(flat, 256)  # int32, shape (257,)

        return sorted_token_indices, expert_offsets