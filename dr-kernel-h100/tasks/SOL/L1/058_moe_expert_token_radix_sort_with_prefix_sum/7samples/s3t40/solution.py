import torch
import triton
import triton.language as tl


@triton.jit
def _stable_rank_argsort_indices(a_ptr, N, out_ptr):
    """
    Compute stable argsort permutation of 'a' (length N).
    For each original index i, compute its rank via stable comparison with all j,
    then write i into out[rank]. Returns a 1D int32 tensor of length N.
    """
    i = tl.program_id(0)  # one program per original index
    # Bounds check for safety (though grid is N)
    if i >= N:
        return

    # Load the value at position i
    val_i = tl.load(a_ptr + i)

    # Initialize rank for this element
    rank = tl.zeros((), dtype=tl.int32)

    # Scan all j to compute stable rank
    # Use a for-loop over j in [0, N). Triton will compile it; this is acceptable for moderate N.
    for j in range(0, N):
        val_j = tl.load(a_ptr + j)
        # less if strictly smaller
        less = val_j < val_i
        # tie if equal values and j < i for stable order
        tie = (val_j == val_i) & (j < i)
        # accumulate rank: number of elements less than val_i + number of earlier equals
        rank += less.to(tl.int32) + tie.to(tl.int32)

    # Write i to out[rank] (out must have length N, int32)
    tl.store(out_ptr + rank, i)


@triton.jit
def _histogram_kernel(vals_ptr, N, hist_ptr, num_buckets: tl.constexpr):
    """
    Count occurrences of each value in vals_ptr (int32) into hist_ptr (int32).
    Assumes vals_ptr[i] in [0, num_buckets-1]. Performs one atomic_add per element.
    """
    pid = tl.program_id(0)  # grid size >= N
    if pid >= N:
        return
    val = tl.load(vals_ptr + pid)  # int32
    # Atomic add to the corresponding bucket
    tl.atomic_add(hist_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(inp_ptr, out_ptr, num_buckets: tl.constexpr):
    """
    Inclusive prefix sum of a small array (num_buckets is small, e.g., 256).
    Compute out[i+1] = sum_{k=0..i} inp[k], out[0] must be set by host code.
    """
    total = tl.zeros((), dtype=tl.int32)
    # Loop over buckets sequentially; num_buckets is constexpr, Triton will unroll
    for i in range(0, num_buckets):
        v = tl.load(inp_ptr + i)  # int32
        total += v
        tl.store(out_ptr + i + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Computes sorted_token_indices (argsort stable of flattened values).
        - Computes expert_offsets (cumulative count of each expert ID).
        All heavy computations are done in Triton kernels. Host code only allocates and launches kernels.
        """
        # Ensure contiguous 1D int32 flat values on the same device as input
        flat = topk_idx.contiguous().view(-1).to(torch.int32)

        N = flat.numel()
        device = flat.device

        # 1) Stable argsort via Triton: out[i] = index of the i-th smallest original element
        out = torch.empty(N, dtype=torch.int32, device=device)
        # Launch one program per original index i
        grid_argsort = (N,)
        _stable_rank_argsort_indices[grid_argsort](flat, N, out)

        # 2) Histogram of expert IDs using Triton
        num_experts = 256  # matches original code's num_experts
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert_offsets (inclusive)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # Set starting offset at 0 for bucket 0
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
