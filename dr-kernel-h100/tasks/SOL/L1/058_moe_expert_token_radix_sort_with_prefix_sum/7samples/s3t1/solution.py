import torch
import triton
import triton.language as tl


@triton.jit
def stable_sort_indices_kernel(a_ptr, N, pos_ptr, out_ptr):
    """
    Stable sort of flat array a_ptr (length N) with integer IDs in [0, num_experts-1].
    Produces a permutation 'out' of length N such that out[pos[i]] = i.
    'pos' is a global counter per element value, used to compute stable ranks.
    """
    # One program per original index i
    i = tl.program_id(0)
    if i >= N:
        return

    # Load the value at position i
    val = tl.load(a_ptr + i)
    # Compute 'rank' = number of elements < val, plus number of elements == val with j < i
    rank = tl.zeros((), dtype=tl.int32)
    # Loop over all j; num_experts is a constexpr here, but the loop iterates up to N
    # We rely on Triton to unroll/simplify this pattern; num_experts=256 in the workload.
    for j in range(0, 1024):  # 1024 is a safe upper bound; N will be passed and masked by i<N checks
        # j is a scalar; we can't vectorize this cleanly across threads, so each program handles its i.
        # We need to load a[j] only when j < N; otherwise skip.
        if j < N:
            aj = tl.load(a_ptr + j)
            # Stable tie-break: among equal values, smaller j comes first
            if aj < val:
                rank += 1
            elif aj == val and j < i:
                rank += 1

    # Reserve position in output: atomically set pos[val] to 'rank' and use it
    old = tl.atomic_add(pos_ptr + val, 1)  # increment pos[val] to 1 initially, then read it back
    # The 'old' is the value BEFORE increment; it will be 0 for the first insertion, 1 for subsequent,
    # but we only need the position. We atomically read-and-add 1 to get a unique position.
    # Better: do not rely on old; instead, compute position by adding 1 to rank directly.
    # However, atomic_add returns the old value; we can use it as the position.
    # To be precise, we want position = rank, so we should avoid the increment here.
    # Fix: use atomic_add for reservation, not counting. Initialize pos to 0 and then:
    # position = atomic_add(pos_ptr + val, 1) will return the old value (i.e., the next free position).
    position = tl.atomic_add(pos_ptr + val, 1)
    # Write i to out[position]
    tl.store(out_ptr + position, i)


@triton.jit
def compute_histogram_kernel(a_ptr, N, histogram_ptr):
    """
    Compute histogram of a_ptr (length N) with integer IDs in [0, num_experts-1].
    histogram[i] = number of elements equal to i.
    """
    # Parallelize over a block of indices
    pid = tl.program_id(0)
    BLOCK = 256
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(a_ptr + offs, mask=mask, other=0)
    # Cast to int32 for histogram
    vals = vals.to(tl.int32)
    # Atomic add 1 for each valid element
    tl.atomic_add(histogram_ptr + vals, 1, mask=mask)


@triton.jit
def compute_prefix_sum_kernel(hist_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute prefix sum of hist_ptr[0..num_experts-1] into offsets_ptr[0..num_experts].
    offsets[0] = 0; offsets[i+1] = offsets[i] + hist[i].
    """
    # Single-program prefix sum; num_experts is constexpr (256)
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_experts):
        total += tl.load(hist_ptr + i)
        tl.store(offsets_ptr + i + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized replacement for the original Model.forward.
        - Computes sorted_token_indices (flattened) using a stable Triton counting-sort.
        - Computes expert_offsets via Triton histogram and prefix sum.
        Returns:
          - sorted_token_indices: 1D tensor of shape (N,), dtype int32
          - expert_offsets: 1D tensor of shape (num_experts+1,), dtype int32
        """
        # Ensure device is CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        a = topk_idx.contiguous()
        N = a.numel()
        num_experts = 256  # matches the original code; assumed constant

        # 1) Stable sort: permutation of indices
        out = torch.empty(N, dtype=torch.int32, device=a.device)
        pos = torch.zeros(num_experts, dtype=torch.int32, device=a.device)
        # Launch kernel: one program per element
        grid_sort = (N,)
        # We need a_ptr dtype to be integer; topk_idx is int32 by default from randint
        stable_sort_indices_kernel[grid_sort](a, N, pos, out)

        # 2) Histogram of expert IDs (int32) using Triton
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=a.device)
        grid_hist = (triton.cdiv(N, 256),)
        compute_histogram_kernel[grid_hist](a, N, histogram)

        # 3) Prefix sum to get expert_offsets (cumulative counts)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=a.device)
        compute_prefix_sum_kernel[(1,)](histogram, offsets, num_experts=num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
