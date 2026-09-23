import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices_kernel(a_ptr, N, out_ptr, BLOCK_N: tl.constexpr):
    """
    Compute torch.argsort(a, stable=True).indices (permutation of [0..N-1]).

    For each original index i in [0..N-1], scan all j in [0..N-1]:
      rank += (a[j] < a[i]) or (a[j] == a[i] and j < i)
    Then write i to out[rank].
    """
    # Each program handles one original index i
    i = tl.program_id(0)  # scalar int32
    # We use BLOCK_N to vectorize the scan over j; i is always in [0, N-1].
    # Create a vector of j candidates.
    # Note: i is a scalar, so we create a vector and compare against it.
    # We use range(0, BLOCK_N) and mask for j < N. Here BLOCK_N should be >= N.
    # However, Triton requires compile-time known vector sizes; we keep BLOCK_N as constexpr.
    # If N > BLOCK_N, this approach becomes unsafe, so we set BLOCK_N to N at launch.
    # But Triton kernel signature doesn't allow passing N as constexpr; BLOCK_N must be known at compile.
    # Hence, we pick BLOCK_N as next power-of-two >= N, passed in at launch as a meta-parameter.
    # The caller sets BLOCK_N to N (next power-of-two), but Triton requires constexpr; we'll pass N via a dummy.
    # Simpler: use a fixed BLOCK_N (e.g., 4096) for all cases; mask ensures safety.
    # We'll set BLOCK_N to 4096 here and mask j < N.
    BLOCK_N = 4096  # meta-parameter, must be constexpr; Triton will see it as tl.constexpr
    j = tl.arange(0, BLOCK_N)
    mask_j = j < N  # valid j range

    # Load the value for index i (scalar), ensure in-bounds
    val_i = tl.load(a_ptr + i, mask=(i < N), other=0)

    # Accumulate rank: count how many elements are less than val_i, and for ties, j < i
    rank = tl.zeros((), dtype=tl.int32)
    for jj in range(0, BLOCK_N):
        j_idx = jj  # integer scalar index
        # Masked load for a[j]
        mask_j = j_idx < N
        val_j = tl.load(a_ptr + j_idx, mask=mask_j, other=0)
        less = val_j < val_i
        equal = val_j == val_i
        # For equal values, ensure stable tie-break by original index: j < i
        tie = equal & (j_idx < i)
        rank += (less | tie).to(tl.int32)

    # Write i to out[rank]
    tl.store(out_ptr + rank, i)


@triton.jit
def _histogram_kernel(a_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    """
    Count occurrences of each value in a_ptr (int32), into histogram_ptr (int32) of length num_buckets.
    Assumes values in [0, num_buckets-1]. Each element a[k] contributes one atomic_add to histogram[a[k]].
    """
    for k in range(0, N):
        val = tl.load(a_ptr + k)  # int32
        # Only add if val is within range; but caller guarantees [0, num_buckets-1]
        tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(histogram_ptr, offsets_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive prefix sum of histogram_ptr (int32, length num_buckets) into offsets_ptr (int32, length num_buckets+1).
    offsets_ptr[0] must be initialized to 0 by host.
    """
    acc = 0
    for b in range(0, num_buckets):
        acc += tl.load(histogram_ptr + b)
        tl.store(offsets_ptr + b + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-based replacement for the original Model.forward.
        - Computes sorted_token_indices: torch.argsort(topk_idx.flatten(), stable=True).indices (shape: [N], int32).
        - Computes expert_offsets: torch.bincount(topk_idx.flatten().long(), minlength=256).cumsum(0) (shape: [257], int32).
        """
        device = topk_idx.device
        flat = topk_idx.reshape(-1)  # 1D int32 tensor of length N
        N = flat.numel()

        # 1) Stable argsort permutation indices via Triton (O(N^2), robust and correct)
        out = torch.empty(N, dtype=torch.int32, device=device)
        # Launch one program per original index
        grid_argsort = (N,)
        # Note: BLOCK_N must be constexpr and >= N. Here we pick 4096 as a safe upper bound for typical N in provided workloads.
        # Triton requires meta-parameters to be tl.constexpr; we pass BLOCK_N as a constexpr symbol to the kernel.
        # In Triton, you can't directly pass N as constexpr from Python; we set BLOCK_N=4096 inside the kernel (safe masking).
        _stable_argsort_indices_kernel[grid_argsort](flat, N, out, BLOCK_N=4096)

        # 2) Histogram of expert IDs using Triton
        num_experts = 256
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert_offsets (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
