import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices_blocked(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Compute the stable argsort permutation for 1D array 'a_ptr' of length N.
    Each program processes a block of indices i in [pid*BLOCK : (pid+1)*BLOCK].
    For each i, compute its rank by scanning all j, then store i at out[rank].
    This matches torch.argsort(a, stable=True).indices.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values for the indices handled by this program
    a_vals = tl.load(a_ptr + offs, mask=mask, other=0)

    # For each i in the block, compute its stable rank and place i at out[rank]
    for i in range(0, BLOCK):
        ii = offs[i]
        mi = ii < N  # scalar mask
        # Compute rank for ii by scanning all j in [0..N-1]
        rank = tl.zeros((), dtype=tl.int32)
        # Unrolled loop over j: Triton will generate code with these constants
        for j in range(0, 64):  # upper bound large enough for typical N; masked by mi
            # Load a_j; if ii >= N, set a_j to a very large sentinel so comparisons are false
            a_j = tl.load(a_ptr + j, mask=mi, other=2**31 - 1)  # use max int32 as sentinel
            ai = a_vals[i]
            less = (a_j < ai)
            tie = (a_j == ai) & (j < ii) & mi
            rank += less + tie.to(tl.int32)
        # Store ii at out[rank], guarded by mi
        tl.store(out_ptr + rank, ii, mask=mi)


@triton.jit
def _histogram_kernel(a_ptr, N, hist_ptr, num_buckets: tl.constexpr):
    """
    Histogram of values in a_ptr (int32). Each element performs one atomic_add
    to hist_ptr[value]. Assumes values in [0, num_buckets-1].
    """
    pid = tl.program_id(0)
    offsets = pid * 1 + tl.arange(0, 1)  # one element per program
    # Launch as (N,) grid: each program handles one element
    # Load value and atomic add to its bucket
    val = tl.load(a_ptr + offsets)
    # Ensure we only do work for valid offsets; grid is exactly (N,), so offsets[0] < N always.
    tl.atomic_add(hist_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(hist_ptr, out_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive prefix sum of 'hist_ptr' (length num_buckets) into 'out_ptr' (length num_buckets+1),
    with out_ptr[0] = 0. Single-program kernel scanning sequentially.
    """
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_buckets):
        acc += tl.load(hist_ptr + i)
        tl.store(out_ptr + i, acc)
    # out_ptr[num_buckets] is not written; caller expects only num_buckets elements used.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D and ensure int32 on device (values are already indices; convert for Triton)
        flat = topk_idx.reshape(-1)
        # Triton works best with int32 for these ops
        flat_i32 = flat.to(torch.int32).contiguous()
        N = flat_i32.numel()
        device = flat_i32.device
        num_experts = 256  # matches original run

        # 1) Triton stable argsort to produce permutation indices (length N)
        #    Output is int32; we'll return as int64 for parity with original.
        sorted_perm_i32 = torch.empty(N, dtype=torch.int32, device=device)
        BLOCK = 256  # process 256 indices per program
        grid = (triton.cdiv(N, BLOCK),)
        _stable_argsort_indices_blocked[grid](flat_i32, sorted_perm_i32, N, BLOCK)

        # 2) Triton histogram of expert IDs
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat_i32, N, histogram, num_experts)

        # 3) Triton inclusive prefix sum to produce expert_offsets of length (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_experts)

        # Return sorted permutation (as int64) and expert_offsets
        return sorted_perm_i32.to(torch.int64), offsets