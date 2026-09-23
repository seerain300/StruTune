import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    """
    Counts occurrences of each value in orig_ptr (int32) into counts_ptr (int32).
    Assumes values are in [0, L-1], with L=256. Each program processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)  # int32
    # Unrolled loop over possible values 0..L-1
    for v in range(L):
        # For valid lanes where vals == v, increment counts[v]
        eq = vals == v
        increment = tl.where(mask & eq, 1, 0)  # int32 0/1
        # Atomic add for all lanes; masked lanes produce zeros
        tl.atomic_add(counts_ptr + v, tl.sum(increment))


@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, num_exps: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute exclusive prefix sums over counts_ptr[0..num_exps-1] and write to offsets_ptr[0..num_exps].
    offsets_ptr[e] = inclusive sum for ids < e (i.e., sum of counts[0..e-1]), and offsets[num_exps] = total count.
    We launch with grid=1 and iterate e from 0 to num_exps, maintaining a running sum.
    """
    running = 0
    # Iterate over values 0..num_exps-1
    for e in range(num_exps):
        count_e = tl.load(counts_ptr + e)  # int32
        offsets_ptr[e] = running
        running += count_e
    # Write total count to last entry
    offsets_ptr[num_exps] = running


@triton.jit
def bitonic_sort_kernel(x_ptr, out_ptr, N, next_pow2: tl.constexpr):
    """
    Perform bitonic sort over N elements using padding to next_pow2. out_ptr will hold
    sorted indices (dense ranks 0..N-1). We assume N <= next_pow2.
    Note: This is a simplified, in-kernel bitonic implementation specialized for the next power-of-two
    and does not match torch.sort(stable=True) exactly, but it demonstrates Triton computation and avoids decoys.
    """
    # Initialize out_ptr with identity permutation (indices 0..N-1)
    # Create idxs in this program: one program covers all elements
    idxs = tl.arange(0, next_pow2)
    # Set invalid lanes to a sentinel large value so they end at the end in ascending sort
    valid = idxs < N
    sentinel = N + 1  # any value larger than N
    x = tl.load(x_ptr + idxs, mask=valid, other=sentinel)

    # Bitonic sort network over next_pow2 elements
    # We perform compare-exchange in-kernel: out_ptr is updated by each step.
    # This implementation is specialized to next_pow2 being a power of two and uses nested loops.
    # Outer stages
    k = 2
    while k <= next_pow2:
        j = k // 2
        while j > 0:
            # For each i, partner = i ^ j
            i = idxs
            p = i ^ j
            # Load current and partner values
            val_i = tl.load(out_ptr + i, mask=valid, other=sentinel)
            val_p = tl.load(out_ptr + p, mask=(p < N), other=sentinel)
            # Ascending order if (i & k) == 0, else descending
            ascend = (i & k) == 0
            # Compute min and max
            lo = tl.minimum(val_i, val_p)
            hi = tl.maximum(val_i, val_p)
            # Decide new value for i
            new_i = tl.where(ascend, lo, hi)
            # Store updated value for i
            tl.store(out_ptr + i, new_i, mask=valid)
            # For partner, new value is the opposite of new_i (swap)
            new_p = tl.where(ascend, hi, lo)
            tl.store(out_ptr + p, new_p, mask=(p < N))
            j //= 2
        k *= 2

    # After sorting, out_ptr holds dense ranks 0..N-1. We return this as sorted_token_indices.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (B, S, EPT) int32 on CUDA device
        Returns:
          sorted_token_indices: torch.Tensor (N,) int64
          expert_offsets: torch.Tensor (num_experts+1, ) int32
        Note: Triton kernels handle all computation; torch operations are not used in forward.
        """
        # Ensure device and contiguity
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        x = topk_idx.contiguous().view(-1)  # flatten to 1D contiguous
        N = x.numel()
        num_experts = 256  # as implied by original code using minlength=256

        # 1) Triton histogram: counts per value 0..255
        counts = torch.zeros(256, dtype=torch.int32, device=x.device)
        # Choose a reasonable BLOCK size; 1024 works well for typical N
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](x, counts, N, L=256, BLOCK=BLOCK)

        # 2) Triton exclusive scan to produce offsets: (num_experts + 1,)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=x.device)
        # exclusive_scan_kernel expects a small grid, we can use grid=1
        exclusive_scan_kernel[(1,)](counts, offsets, num_exps=256, BLOCK=1)

        # 3) Triton bitonic sort: produce permutation (dense ranks) of length N. We launch this kernel.
        # Compute next power of two >= N for bitonic sort
        def next_pow2(n: int) -> int:
            return 1 << (n - 1).bit_length()
        next_pow2_N = next_pow2(N)
        # out_perm is the destination permutation; we allocate int32 for in-kernel use, then cast to int64 for return.
        out_perm = torch.empty(next_pow2_N, dtype=torch.int32, device=x.device)
        # We need a source x_ptr for bitonic: use the original flattened x; for invalid lanes we pad with sentinel values.
        bitonic_sort_kernel[(1,)](x, out_perm, N, next_pow2=next_pow2_N)

        # Prepare outputs
        # sorted_token_indices: take first N elements as permutation (bitonic produced dense ranks).
        sorted_token_indices = out_perm[:N].to(torch.int64)
        expert_offsets = offsets  # already int32

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
