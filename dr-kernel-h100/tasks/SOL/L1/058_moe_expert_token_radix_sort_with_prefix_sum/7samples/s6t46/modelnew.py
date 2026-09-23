import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: Global stable argsort via counting sort for values in [0, 255].
# Inputs:
#   flat: 1D int32 values
#   out_idx: 1D int64 output permutation (length N)
#   scan: 1D int32 prefix sum array of per-class counts (length 257), where scan[0]=0 and scan[1..]=inclusive counts
#   N: total number of elements in flat
#   NUM_CLASSES: compile-time constant 256
@triton.jit
def _global_argsort_stable_kernel(flat, out_idx, scan, N, NUM_CLASSES: tl.constexpr):
    # We use a single program with a serial loop per class, which is fine for the provided N sizes.
    for c in range(NUM_CLASSES):
        start = scan[c]  # inclusive start index for class c
        # First pass: write each i where flat[i] == c at out_idx[start++]
        for i in range(0, N):
            if flat[i] == c:
                tl.store(out_idx + start, i)  # store index as int64 (Triton will handle int32/64 store)
                start += 1
        # Second pass: pad with zeros if this class has more slots than actual count
        # Note: We cannot branch on runtime N inside Triton without dynamic control flow; but here we know
        # scan[c+1] - scan[c] is the required number of elements for class c. After the first pass, start
        # equals the number of actual elements. If scan[c+1] > start, we need to pad with zeros to reach scan[c+1].
        # Triton doesn't support dynamic 'while' here; the simplest is to iterate up to N and only write when idx < scan[c+1].
        end = scan[c + 1]
        for i in range(0, N):
            if start < end:
                tl.store(out_idx + start, 0)
                start += 1
            else:
                break


# Kernel: Compute per-class counts for flat (int32 values in [0, NUM_CLASSES-1]) using atomic adds.
# Output counts: int32, length NUM_CLASSES
@triton.jit
def _hist_kernel(flat, counts, N, NUM_CLASSES: tl.constexpr):
    for i in range(0, N):
        c = flat[i]
        # Ensure c is within [0, NUM_CLASSES-1]
        # Triton atomic_add expects int32 pointers and int32 values
        tl.atomic_add(counts + c, 1)


# Kernel: Inclusive scan (prefix sum) of counts using a simple serial loop.
# Input counts: int32, length NUM_CLASSES (256)
# Output scan: int32, length NUM_CLASSES+1 (257), with scan[0]=0, scan[1..]=inclusive prefix sums
@triton.jit
def _inclusive_scan_kernel(counts, scan, NUM_CLASSES: tl.constexpr):
    # scan[0] = 0
    tl.store(scan + 0, 0)
    running = 0
    for i in range(0, NUM_CLASSES):
        running += counts[i]
        tl.store(scan + 1 + i, running)


def _launch_histogram(flat: torch.Tensor) -> torch.Tensor:
    # Compute counts per class via Triton
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Ensure int32 flat for histogram
    flat32 = flat.to(torch.int32)
    N = flat32.numel()
    # Launch histogram kernel (single program with serial loops). Given N sizes in workloads, this is fine.
    _hist_kernel[(1,)](flat32, counts, N, 256)
    return counts


def _compute_inclusive_scan(counts: torch.Tensor) -> torch.Tensor:
    # Compute inclusive scan via Triton
    scan = torch.empty(257, dtype=torch.int32, device=counts.device)
    _inclusive_scan_kernel[(1,)](counts, scan, 256)
    return scan


def _launch_global_argsort(flat: torch.Tensor) -> torch.Tensor:
    # Launch global stable argsort kernel; output is int64
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
    # Note: Triton kernels here run in forward; if Triton is unavailable, this path would not be used in this harness.
    _global_argsort_stable_kernel[(1,)](flat.to(torch.int32), out_idx, None, N, 256)
    return out_idx


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch_size, seq_len, num_experts_per_tok)
        assert topk_idx.dim() == 3, "topk_idx must be 3D: (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D and make contiguous
        flat = topk_idx.reshape(-1).contiguous()
        # Compute sorted_token_indices via Triton global stable argsort (int64)
        sorted_token_indices = _launch_global_argsort(flat)
        # Compute expert_offsets via Triton histogram + inclusive scan (int32, length num_experts+1 = 257)
        expert_offsets = _compute_inclusive_scan(_launch_histogram(flat))
        # Return with exact shapes/dtypes:
        # sorted_token_indices: shape (N,), dtype int64
        # expert_offsets: shape (257,), dtype int32
        return sorted_token_indices, expert_offsets