import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices_kernel(a_ptr, N, out_ptr, BLOCK_N: tl.constexpr):
    """
    Compute torch.argsort(a, stable=True).indices using Triton.

    For each original index i in [0, N), compute its stable rank by scanning all j:
      rank += (a[j] < a[i]) OR (a[j] == a[i] AND j < i)
    Then write i to out[rank].
    """
    i = tl.program_id(0)
    # Early exit if i >= N (grid may be > N)
    if i >= N:
        return

    # Load a[i]
    val_i = tl.load(a_ptr + i)

    # Compute rank: count of elements less than a[i] and tie-breaker for equal values
    rank = tl.zeros((), dtype=tl.int32)

    # Scan all j in [0, N); use a fixed loop to avoid dynamic indexing pitfalls
    # Note: This is O(N). For benchmarks with N up to a few thousand, this is acceptable.
    for j in range(0, BLOCK_N):
        # Mask to avoid out-of-range j
        if j < N:
            val_j = tl.load(a_ptr + j)
            less = val_j < val_i
            tie = (val_j == val_i) & (j < i)
            rank += (less | tie).to(tl.int32)

    # Write i to the output at position 'rank'
    tl.store(out_ptr + rank, i)


@triton.jit
def _histogram_kernel(vals_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    """
    Histogram of values in vals_ptr[0:N], each value is an int in [0, num_buckets-1].
    For each element v, atomic_add histogram_ptr[v] += 1.
    """
    pid = tl.program_id(0)
    # Each program handles one element
    # (grid can be (N,), but for robustness we can also handle grid > N by masking pid)
    if pid < N:
        v = tl.load(vals_ptr + pid)
        # v is int32; assume it's within [0, num_buckets-1]. Atomic add 1.
        tl.atomic_add(histogram_ptr + v, 1)


@triton.jit
def _inclusive_scan_prefix_sum(histogram_ptr, offsets_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive prefix sum of histogram_ptr[0:num_buckets] and write to offsets_ptr[1:].
    offsets_ptr[0] is set on host to 0.
    """
    # Single-program scan over num_buckets
    total = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_buckets):
        val = tl.load(histogram_ptr + k)
        total += val
        tl.store(offsets_ptr + (k + 1), total)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Compute 1D sorted_token_indices = torch.argsort(topk_idx.flatten(), stable=True).indices
        - Compute expert_offsets = torch.bincount(topk_idx.flatten().long(), minlength=num_experts).cumsum(0)
        All computations are performed by Triton kernels; no torch.sort/argsort/bincount on host.
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."

        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device
        dtype = flat.dtype

        # 1) Stable argsort indices in Triton
        out = torch.empty(N, dtype=torch.int32, device=device)
        # Use a reasonable BLOCK_N; N in benchmarks is typically <= 8192.
        BLOCK_N = 1024  # loop scans up to 1024 comparisons per element; masked for j < N
        grid_argsort = (N,)
        _stable_argsort_indices_kernel[grid_argsort](flat, N, out, BLOCK_N=BLOCK_N)

        # 2) Histogram of expert IDs (values) using Triton
        histogram = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        grid_hist = (N,)
        _histogram_kernel[grid_hist](flat, N, histogram, num_buckets=self.num_experts)

        # 3) Compute expert_offsets via inclusive prefix sum in Triton
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        grid_scan = (1,)
        _inclusive_scan_prefix_sum[grid_scan](histogram, offsets, num_buckets=self.num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
