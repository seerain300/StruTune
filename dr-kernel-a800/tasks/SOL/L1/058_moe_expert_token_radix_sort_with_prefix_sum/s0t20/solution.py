import torch
import triton
import triton.language as tl


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, idx_out_ptr, N, LOGN: tl.constexpr):
    """
    Perform stable bitonic sort of flat_ptr (int32) values and output sorted positions
    in idx_out_ptr (int64). Grid: axis=0 = N programs, axis=1 = LOGN stages.
    For each stage j (j in [1, LOGN-1]), each program i:
      - partner = i ^ (1 << j)
      - asc = ((i & (1 << (j+1))) == 0)
      - tie-break: if equal, lower index comes first (i < partner)
      - load value v_i = tl.load(flat_ptr + idx_out_ptr[i]), v_p
      - compute min/max and decide new position based on asc and tie
      - write to a temporary idx_out_ptr for this stage; next stage reads from updated idx_out_ptr
    This approach avoids Python bitwise ops on Triton tensors and uses vectorized compare-and-swap.
    """
    pid = tl.program_id(axis=0)
    # For each stage j, update idx_out_ptr to contain new positions based on partner comparisons.
    # We'll loop j from 1 to LOGN-1 using Triton's constexpr. For each j, recompute partner for pid.
    # Note: Since Triton kernels are compiled, passing LOGN as tl.constexpr allows unrolling of loops.

    # The kernel operates in-place on idx_out_ptr by updating it each iteration. We don't need an
    # auxiliary buffer; Triton allows in-place updates per program. However, implementing multi-stage
    # updates requires reading partner's position at each stage. Triton supports such operations via
    # tl.load/tl.store and masks.

    # For clarity, we implement the full bitonic network using the standard algorithm:
    # For j in [1, LOGN-1]:
    #   partner = pid ^ (1 << j)
    #   asc = ((pid & (1 << (j+1))) == 0)
    #   tie = (v_i == v_p)
    #   lower = (pid < partner)
    #   if asc:
    #       swap if v_i > v_p or (equal and not lower)
    #   else:
    #       swap if v_i < v_p or (equal and not lower)
    # We compute new_pos for pid based on these conditions and write it back to idx_out_ptr[pid].
    # Since we cannot branch on partner's index, we compute new_pos using pid's own index; partner's
    # index update is handled by the next pid that has partner as its own index (i.e., both update
    # simultaneously). In Triton, each program instance runs independently; to make this work,
    # we update idx_out_ptr in each iteration and rely on consistent data after j iterations.

    for j in range(1, LOGN):
        partner = pid ^ (1 << j)
        # Determine ascending/descending for this stage
        asc = ((pid & (1 << (j + 1))) == 0)
        # Load current positions and values
        pos_i = tl.load(idx_out_ptr + pid)  # current position (int64)
        pos_p = tl.load(idx_out_ptr + partner)  # partner position (int64)
        # Load values at those positions from flat_ptr
        v_i = tl.load(flat_ptr + pos_i)  # int32
        v_p = tl.load(flat_ptr + pos_p)  # int32
        # Compute min/max
        minv = tl.minimum(v_i, v_p)
        maxv = tl.maximum(v_i, v_p)
        # Stability tie-break: lower index comes first if equal
        tie = (v_i == v_p)
        lower = (pid < partner)
        # Decide whether to swap
        swap = False
        if asc:
            # Ascending: swap if v_i > v_p or (equal and not lower)
            swap = (v_i > v_p) | ((v_i == v_p) & (~lower))
        else:
            # Descending: swap if v_i < v_p or (equal and not lower)
            swap = (v_i < v_p) | ((v_i == v_p) & (~lower))
        # Compute new position for pid
        new_pos = tl.zeros((), dtype=tl.int64)
        if swap:
            new_pos = pos_p
        else:
            new_pos = pos_i
        # Write back to idx_out_ptr[pid]
        tl.store(idx_out_ptr + pid, new_pos)

    # After LOGN-1 stages, idx_out_ptr contains the sorted positions corresponding to flat_ptr.
    # We return idx_out_ptr; since this is an in-place update, the final output is idx_out_ptr.


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    pid = tl.program_id(axis=0)
    # Single program computes the prefix sum sequentially
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Inputs: topk_idx of shape (B, S, EPT), int32, on CUDA.
        Outputs:
          sorted_token_indices: int64, shape (N,), sorted positions corresponding to flat.
          expert_offsets: int32, shape (num_experts + 1,), inclusive prefix sums of counts.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        flat = topk_idx.reshape(-1)  # int32 on CUDA
        N = flat.numel()
        num_experts = 256
        # Compute LOGN = ceil(log2(N)) for bitonic sort. Triton requires constexpr.
        # We set LOGN for N up to 8192 (13 stages). For the provided workloads, N is much smaller.
        # If N is larger, you can increase LOGN accordingly (e.g., up to 14 or 15). Here we use 13.
        # Note: We need LOGN as tl


def run(*args):
    return ModelNew()(*args)
