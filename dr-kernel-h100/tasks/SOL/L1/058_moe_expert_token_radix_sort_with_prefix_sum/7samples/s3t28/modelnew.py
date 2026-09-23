import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_by_values_kernel(flat_ptr, out_ptr, N: tl.int32, num_experts: tl.constexpr):
    """
    Compute stable argsort permutation 'out' of length N based on values in 'flat'.
    For each original index i in [0, N), out[i] is the index of the i-th smallest element of 'flat',
    with stable tie-breaking: equal values preserve original order (smaller original index first).
    """
    # Process one bucket per program to keep things simple and stable.
    # We'll loop over all buckets b in [0, num_experts) and handle insertion for each.
    # Each program (pid=0..num_experts-1) handles bucket b and writes positions for all indices i where flat[i] == b.
    # Note: num_experts is constexpr, so Triton will unroll or loop safely.
    for b in range(num_experts):
        # Running start position in the output for this bucket (number of elements already inserted with j < i).
        start = tl.zeros((), dtype=tl.int32)
        # Iterate i = 0..N-1, assign stable positions for flat[i] == b
        for i in range(N):
            val = tl.load(flat_ptr + i)  # int32
            is_equal = val == b
            # Stable rank among equal values: number of elements j < i with flat[j] == b
            rank = start
            for j in range(i):
                vj = tl.load(flat_ptr + j)
                # Only count if j < i and equal to current bucket
                count_j = (vj == b) & (j < i)
                rank += count_j.to(tl.int32)
            # Place i at out[rank]
            tl.store(out_ptr + rank, i, mask=is_equal)
            # Update start for next equal element: after inserting i at rank, start increases by 1 when we saw i.
            # We need to increment start only if i was inserted (i.e., we took the mask path). Triton doesn't have a
            # conditional side-effect for store, so we guard the increment by is_equal.
            if is_equal:
                start += 1


@triton.jit
def _histogram_kernel(flat_ptr, out_ptr, N: tl.int32, num_buckets: tl.constexpr):
    """
    Compute histogram of values in 'flat' into 'out' (length num_buckets).
    out[b] = number of elements equal to b.
    """
    for i in range(N):
        val = tl.load(flat_ptr + i)
        # Atomic add 1 for each occurrence of val
        tl.atomic_add(out_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(in_ptr, out_ptr, n: tl.int32):
    """
    Compute inclusive scan (prefix sum) of 'in_ptr[0..n-1]' into 'out_ptr[1..n]', out_ptr[0] is left untouched.
    Uses iterative doubling: each thread handles positions spaced by 2^k and adds in[i] from i - 2^k when valid.
    Assumes 'in_ptr' and 'out_ptr' are int32 and 'n' is known at launch time.
    """
    # We operate with positions: p = 0..n-1
    # For each k, each thread i updates out_ptr[i] += in_ptr[i - 2^k] if valid.
    # We implement this as a simple loop over k with a per-thread mask.
    # Note: Triton encourages block processing; here we do per-element masks.
    # Start with in_ptr values as 0..n-1 indices then load values from memory? Instead, we directly
    # do the update in-place via pointer arithmetic. Triton allows pointer-based math for such scans.
    # Implementation: perform scan in-place using masked adds; we'll read current out_ptr[p] and add shifted values.
    # This is a standard parallel inclusive scan pattern. We assume 'out_ptr[0..n-1]' is already initialized to 0.
    # We'll iterate k from 0 up to the maximum needed. Since n is runtime, we can unroll up to a reasonable max
    # but Triton requires constexpr loops. To handle runtime n, we instead implement a simple sequential scan:
    # out_ptr[1..] = cumulative sum of in_ptr[0..], which is fine for small n (here num_experts=256).
    # However, to be robust, we'll implement the doubling scan using masks with per-thread indices.
    # Triton does not expose a vector range like tl.arange with dynamic n easily; instead, we write a per-thread
    # loop that handles each position p by loading shifted value out_ptr[p - 2^k] when valid.
    # Start: out_ptr[:] = 0
    # Then for k = 0,1,2,... while (1 << k) <= n:
    #   For each p in 0..n-1, if p >= (1 << k), then out_ptr[p] += out_ptr[p - (1 << k)].
    # We implement this via a Python-side for k in range(...) loop; Triton supports Python loops when bounds
    # are static or can be computed. We'll pass n as constexpr by using a constexpr upper bound and mask.
    # Simplify: we set out_ptr[0..] = 0; then run sequential scan with k-loop. Triton allows simple loops.
    # Since Triton doesn't support arbitrary Python while loops cleanly here, we instead implement a sequential
    # scan: out_ptr[1] = in_ptr[0], out_ptr[2] = out_ptr[1] + in_ptr[1], ..., which is fine for n up to 256.
    # This avoids the complexity of parallel scan and guarantees correctness for num_experts=256.

    # First, initialize out[1..] to zeros
    # We cannot directly write a range in Triton easily; but since we compute offsets via atomic_add above,
    # here we assume 'in_ptr' holds the counts for each bucket and compute the prefix sums.
    # We'll do a simple per-element scan:
    total = tl.zeros((), dtype=tl.int32)
    for p in range(n):
        val = tl.load(in_ptr + p)
        total += val
        tl.store(out_ptr + p + 1, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is hard-coded as 256 in the original reference; we mirror that.
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure device and dtype
        device = topk_idx.device
        # Flatten and cast to int32 for Triton
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = flat.numel()

        # 1) Stable argsort permutation using Triton
        out = torch.empty(N, dtype=torch.int32, device=device)  # out[i] = index of i-th smallest element in flat
        grid = (self.num_experts,)  # one program per bucket
        _stable_argsort_by_values_kernel[grid](flat, out, N, num_experts=self.num_experts)

        # 2) Histogram of expert IDs using Triton
        histogram = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(1,)](flat, histogram, N, num_buckets=self.num_experts)

        # 3) Prefix sum to get expert_offsets (length num_experts + 1), starting at 0
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0  # inclusive prefix sums; we'll fill [1..] from histogram
        # Simple sequential scan works and is fine for num_experts=256
        total = 0
        for p in range(self.num_experts):
            total += histogram[p]
            offsets[p + 1] = total

        # Return sorted_token_indices (1D of length N) and expert_offsets (1D of length 257)
        return out, offsets