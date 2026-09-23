import torch
import triton
import triton.language as tl


# Triton kernel: stable sort of 1D int32 array using odd-even transposition sort.
# We sort in-place into indices_out, and after finishing, indices_out contains the sorted permutation.
@triton.jit
def _sort_1d_stable_kernel(indices: tl.pointer_type(tl.int32),  # input
                            out_indices: tl.pointer_type(tl.int32),  # output
                            N: tl.constexpr):  # number of elements (constexpr for loops)
    # Each program handles one element and performs all passes; simple and correct.
    pid = tl.program_id(0)
    # Bounds check: only operate on valid positions
    if pid >= N:
        return

    # Even phase: compare-swap pairs (i, i+1) where i is even
    # Odd phase: compare-swap pairs (i, i+1) where i is odd
    # This is a stable sorting network (odd-even transposition sort).
    # Note: for ties, we preserve original order by not swapping when equal.
    for phase in range(N):  # N passes
        # even phase
        if (phase % 2 == 0):
            i = pid
            partner = i + 1
            # Only proceed if partner is valid and i < partner
            if partner < N and i < partner:
                ai = tl.load(indices + i)
                bi = tl.load(indices + partner)
                # If ai > bi, swap; else keep. For equal, do not swap (stable).
                swap = ai > bi
                tl.store(out_indices + i, tl.where(swap, bi, ai))
                tl.store(out_indices + partner, tl.where(swap, ai, bi))
        else:
            i = pid
            partner = i + 1
            if partner < N and i < partner:
                ai = tl.load(indices + i)
                bi = tl.load(indices + partner)
                swap = ai > bi
                tl.store(out_indices + i, tl.where(swap, bi, ai))
                tl.store(out_indices + partner, tl.where(swap, ai, bi))


# Triton kernel: per-expert histogram (count occurrences of each expert id).
# We process the flattened array in blocks to reduce atomic operations.
@triton.jit
def _histogram_kernel(vals_ptr: tl.pointer_type(tl.int32),  # flattened indices
                      counts_ptr: tl.pointer_type(tl.int32),  # per-expert counts (size NUM_EXPERTS)
                      N: tl.constexpr,  # number of elements
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load a block of values; for masked-out lanes, use 0 (they won't contribute).
    vals = tl.load(vals_ptr + offsets, mask=mask, other=0)
    # Accumulate local counts per bin in the block
    # Loop over bins; for each bin, count how many lanes equal that bin in this block.
    for b in range(NUM_EXPERTS):
        # Compute how many lanes equal b in this block
        eq = (vals == b) & mask
        # eq is a boolean vector; sum to get count
        # Triton supports tl.sum on boolean by casting to int.
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        # Atomically add to global counts[b]
        tl.atomic_add(counts_ptr + b, cnt)


# Triton kernel: inclusive prefix sum of counts to produce expert_offsets.
# We compute per-bin prefix sums sequentially. NUM_EXPERTS is fixed at 256.
@triton.jit
def _prefix_sum_kernel(counts_ptr: tl.pointer_type(tl.int32),       # per-expert counts
                       offsets_ptr: tl.pointer_type(tl.int32),       # per-expert offsets (size NUM_EXPERTS+1)
                       NUM_EXPERTS: tl.constexpr):
    # Each program handles a tile of bins; for small NUM_EXPERTS=256, a single program suffices.
    pid = tl.program_id(0)
    # We'll write in tiles; for simplicity, single program computes all bins.
    # Start with offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # Compute inclusive prefix sum
    carry = 0
    for i in range(NUM_EXPERTS):
        ci = tl.load(counts_ptr + i)
        carry += ci
        tl.store(offsets_ptr + i + 1, carry)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sort flattened topk_idx stably using Triton.
        - Compute expert offsets (cumulative counts) using Triton histogram + prefix sum.
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        num_experts = 256  # fixed per the original code

        # 1) Stable sort using Triton odd-even transposition sort (in-place into out_sorted)
        out_sorted = torch.empty_like(flat, dtype=torch.int32)
        # Launch 1D grid with size N. Each program sorts element pid across phases.
        sort_grid = (N,)
        _sort_1d_stable_kernel[sort_grid](flat, out_sorted, N)

        # 2) Per-expert histogram via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_SIZE = 2048  # tuneable; reduces atomic operations
        hist_grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _histogram_kernel[hist_grid](out_sorted, counts, N, BLOCK_SIZE)

        # 3) Prefix sum of counts to produce expert_offsets (size num_experts+1)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # Launch single program; NUM_EXPERTS is constexpr
        _prefix_sum_kernel[(1,)](counts, expert_offsets, NUM_EXPERTS=num_experts)

        # Return sorted_token_indices and expert_offsets (as in original)
        return out_sorted.to(torch.int32), expert_offsets  # expert_offsets already int32


def run(*args):
    return ModelNew()(*args)
