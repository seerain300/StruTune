import torch
import triton
import triton.language as tl


@triton.jit
def _global_stable_counting_sort_by_class(flat_ptr, out_idx_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Global stable counting sort implemented by per-class sequential assignment.
    For each class c in 0..NUM_CLASSES-1:
      - Initialize class_count = 0
      - Scan all i in 0..N-1
      - If flat[i] == c, write i to out_idx[class_count] and increment class_count
    This preserves original order for equal values (stable).
    Each program handles one class c and scans the entire flat vector deterministically.
    """
    c = tl.program_id(0)
    if c < NUM_CLASSES:
        class_count = tl.zeros((), dtype=tl.int32)
        # Sequentially assign positions for class c
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            is_match = val == c
            if is_match:
                tl.store(out_idx_ptr + class_count, i)
                class_count += 1


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Histogram of flat values into counts_ptr (length NUM_CLASSES).
    Each program handles one class c and scans the flat vector to count occurrences.
    """
    c = tl.program_id(0)
    if c < NUM_CLASSES:
        total = tl.zeros((), dtype=tl.int32)
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            total += (val == c)
        tl.store(counts_ptr + c, total)


@triton.jit
def _inclusive_scan_counts(counts_ptr, inclusive_ptr, M: tl.constexpr):
    """
    Inclusive scan (prefix sum) of 'counts_ptr' of length M, writing results to 'inclusive_ptr'.
    Simple sequential loop in a single program is fine for small M (e.g., 256).
    """
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, M):
        val = tl.load(counts_ptr + i)
        running += val
        tl.store(inclusive_ptr + i, running)


@triton.jit
def _fill_expert_offsets_from_inclusive(inclusive_ptr, offsets_ptr, M: tl.constexpr):
    """
    Fill offsets[1..M] from inclusive prefix sums. offsets[0] must be set by host.
    """
    for k in range(0, M):
        tl.store(offsets_ptr + 1 + k, tl.load(inclusive_ptr + k))


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernel to produce sorted_token_indices = global stable argsort of flat.
    Returns a 1D torch.Tensor of int32 on the same device, shape (N,).
    """
    assert flat.is_cuda, "Triton kernels require CUDA tensors"
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    grid = (256,)  # one program per class
    _global_stable_counting_sort_by_class[grid](flat, out_idx, N, NUM_CLASSES=256)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert offsets via Triton histogram + Triton inclusive scan.
    Returns tensor of shape (num_experts + 1,) int32.
    """
    assert flat.is_cuda, "Triton kernels require CUDA tensors"
    N = flat.numel()
    counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    grid_hist = (num_experts,)
    _hist_kernel[grid_hist](flat, counts, N, NUM_CLASSES=num_experts)
    inclusive = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    _inclusive_scan_counts[(1,)](counts, inclusive, M=num_experts)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[0] = 0
    _fill_expert_offsets_from_inclusive[(1,)](inclusive, offsets, M=num_experts)
    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Computes sorted_token_indices = global stable argsort of flattened topk_idx via Triton.
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