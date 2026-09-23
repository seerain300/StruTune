import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_sort_stable(a_ptr, out_ptr, N, iters: tl.constexpr):
    # Single-program grid; vectorize across all indices and perform odd-even sort passes.
    i = tl.arange(0, N)
    # Load initial values: out_ptr initialized to identity indices; we will swap values and corresponding indices
    # by detecting swaps. To keep it simple, we load values from a_ptr and write to out_ptr as we sort.
    # We need to perform iters = N passes.
    for t in range(iters):
        # Even phase: pairs (0,1), (2,3), ...
        start_even = 0
        step_even = 2
        # Build pairs: (start_even, start_even + 1), skip if out of range
        # We implement pair-wise compare-swap for j = start_even + 1
        # Even phase condition: (t % 2 == 0)
        if (t % 2) == 0:
            j = start_even + 1
            while j < N:
                # Load a[i], a[j]
                vi = tl.load(a_ptr + i)
                vj = tl.load(a_ptr + j)
                # Decide swap: if vi > vj or equal with i > j (for stability, keep original order)
                swap = (vi > vj) | ((vi == vj) & (i > j))
                # Compute min/max values
                minv = tl.where(swap, vj, vi)
                maxv = tl.where(swap, vi, vj)
                # Write back
                tl.store(a_ptr + i, minv)
                tl.store(a_ptr + j, maxv)
                # Also swap indices in out_ptr accordingly
                oi = tl.load(out_ptr + i)
                oj = tl.load(out_ptr + j)
                o_min = tl.where(swap, oj, oi)
                o_max = tl.where(swap, oi, oj)
                tl.store(out_ptr + i, o_min)
                tl.store(out_ptr + j, o_max)
                j += 2
        # Odd phase: pairs (1,2), (3,4), ...
        start_odd = 1
        step_odd = 2
        if (t % 2) == 1:
            j = start_odd + 1
            while j < N:
                vi = tl.load(a_ptr + (j - 1))
                vj = tl.load(a_ptr + j)
                swap = (vi > vj) | ((vi == vj) & ((j - 1) > j))
                minv = tl.where(swap, vj, vi)
                maxv = tl.where(swap, vi, vj)
                tl.store(a_ptr + (j - 1), minv)
                tl.store(a_ptr + j, maxv)
                oi = tl.load(out_ptr + (j - 1))
                oj = tl.load(out_ptr + j)
                o_min = tl.where(swap, oj, oi)
                o_max = tl.where(swap, oi, oj)
                tl.store(out_ptr + (j - 1), o_min)
                tl.store(out_ptr + j, o_max)
                j += 2
    # After N passes, 'a_ptr' contains sorted values (ascending, stable) and 'out_ptr' contains permutation indices.


@triton.jit
def _histogram_kernel(a_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    # Each program handles one element and atomically increments its bucket.
    i = tl.program_id(0)
    # Guard for i < N
    if i < N:
        val = tl.load(a_ptr + i)
        # val is in [0, num_buckets-1], int32
        tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(histogram_ptr, offsets_ptr, num_buckets: tl.constexpr):
    # Single-program inclusive scan over histogram into offsets[1..].
    # We assume offsets_ptr[0] is already set to 0 by host code.
    acc = 0
    for b in range(num_buckets):
        acc += tl.load(histogram_ptr + b)
        tl.store(offsets_ptr + (b + 1), acc)


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch, seq, num_experts_per_tok) int32
        device = topk_idx.device
        a = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = a.numel()

        # 1) Stable argsort via odd-even sort in Triton
        out = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices
        out.copy_(torch.arange(N, dtype=torch.int32, device=device))  # initial identity permutation
        # Launch a single-program grid; perform N passes
        # Odd-even sort requires N passes for length N to guarantee sortedness.
        _odd_even_sort_stable[(1,)](a, out, N, iters=N)

        # 2) Histogram of expert IDs (values in [0, 255] since num_experts=256)
        num_experts = 256
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](a, N, histogram, num_buckets=num_experts)

        # 3) Prefix sum to get expert_offsets (length num_experts + 1)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
