import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute per-class counts for values in flat (int32)
# flat_ptr: *int32, length N
# counts_ptr: *int32, length NUM_CLASSES (256), initialized to zeros
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    # Single program performs linear scan and atomic adds
    # Note: For this small NUM_CLASSES, one program is fine.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        # val is 0..255, ensure int32
        tl.atomic_add(counts_ptr + val, 1)


# Triton kernel: compute inclusive prefix sums (exclusive scan on input counts)
# counts_ptr: *int32, length NUM_CLASSES+1; counts_ptr[0] must be 0
# scan_ptr: *int32, length NUM_CLASSES+1
@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES_PLUS_ONE: tl.int32):
    # This is a simple sequential scan; NUM_CLASSES_PLUS_ONE is small (257).
    acc = 0
    for i in range(0, NUM_CLASSES_PLUS_ONE):
        acc += tl.load(counts_ptr + i)
        tl.store(scan_ptr + i, acc)


# Triton kernel: fill sorted_token_indices permutation
# flat_ptr: *int32, length N
# out_idx_ptr: *int64, length N
# scan_ptr: *int32, length NUM_CLASSES+1
# N: int32
# NUM_CLASSES: int32
@triton.jit
def _global_argsort_stable_kernel(
    flat_ptr, out_idx_ptr, scan_ptr, N: tl.int32, NUM_CLASSES: tl.int32
):
    # Single program performs global stable argsort via counting places
    for c in range(0, NUM_CLASSES):
        start = tl.load(scan_ptr + c)  # inclusive start for class c
        end = tl.load(scan_ptr + c + 1)  # exclusive end
        # iterate over original indices in order; only write when start < end
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            if val == c:
                if start < end:
                    tl.store(out_idx_ptr + start, tl.cast(i, tl.int64))
                    start += 1
                else:
                    # do nothing, class full
                    pass


def _launch_histogram(flat: torch.Tensor) -> torch.Tensor:
    # flat is 1D int32 tensor on device
    N = flat.numel()
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    if TRITON_AVAILABLE:
        _hist_kernel[(1,)](flat, counts, N, 256)
    else:
        # Fallback histogram via torch to ensure correctness if Triton unavailable
        counts = torch.bincount(flat.long(), minlength=256)
    return counts


def _compute_inclusive_scan(counts: torch.Tensor) -> torch.Tensor:
    scan = torch.empty(257, dtype=torch.int32, device=flat.device)
    scan[0] = 0
    if TRITON_AVAILABLE:
        _inclusive_scan_kernel[(1,)](counts, scan, 257)
    else:
        # Manual scan fallback
        acc = 0
        for i in range(257):
            acc += counts[i - 1] if i > 0 else 0
            scan[i] = acc
    return scan


def _launch_global_argsort(flat: torch.Tensor) -> torch.Tensor:
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
    if TRITON_AVAILABLE:
        _global_argsort_stable_kernel[(1,)](flat, out_idx, scan, N, 256)
    else:
        # Fallback: torch.argsort for correctness
        out_idx = flat.argsort(stable=True)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor) -> torch.Tensor:
    # Compute per-class counts and prefix sums using Triton if available
    counts = _launch_histogram(flat)  # int32, length 256
    scan = _compute_inclusive_scan(counts)  # int32, length 257
    # Return offsets[1:], length 256
    return scan[1:]  # int32


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Expect 3D input (batch, seq, num_experts_per_tok)
        assert topk_idx.dim() == 3, "topk_idx must be 3D: (batch_size, seq_len, num_experts_per_tok)"
        # Flatten and ensure contiguous int32 for Triton
        flat = topk_idx.reshape(-1).contiguous()
        flat32 = flat.to(torch.int32)

        # Compute sorted_token_indices via Triton global stable argsort (1D int64)
        sorted_token_indices = _launch_global_argsort(flat32)  # shape: (N,), int64

        # Compute expert_offsets via Triton histogram + scan (1D int32, length num_experts+1 = 257)
        expert_offsets = _compute_expert_offsets(flat32)

        return sorted_token_indices, expert_offsets