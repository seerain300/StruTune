import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Build per-expert counts for flat values in [0, num_experts-1] using atomic adds.
    flat_ptr: *int32, length N
    counts_ptr: *int32, length num_experts
    """
    # Process flat in chunks to avoid overly large while loops
    # We'll iterate with a Python for-loop which Triton supports for scalar i.
    i = 0
    while i < N:
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < N
        vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
        # For each element in the chunk, do atomic add to counts
        # BLOCK is constexpr, so Triton will unroll this loop
        for k in range(BLOCK):
            v = vals[k]
            # only update if valid
            if mask[k]:
                tl.atomic_add(counts_ptr + v, 1)
        i += BLOCK


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr (length num_experts) and write to offsets_ptr (length num_experts+1).
    offsets_ptr[0] = 0, offsets_ptr[i] = offsets_ptr[i-1] + counts_ptr[i-1] for i>0.
    """
    # We can do this sequentially in a single program instance
    # Initialize offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # Compute prefix sum
    carry = 0
    # Loop over i from 0 to num_experts-1 (inclusive sum)
    for i in range(0, num_experts):
        val = tl.load(counts_ptr + i)
        carry += val
        tl.store(offsets_ptr + i + 1, carry)


@triton.jit
def _odd_even_sort_stable(arr_ptr, indices_ptr, N: tl.int32, passes: tl.int32):
    """
    Odd-even transposition sort on arr_ptr (int32) of length N using indices_ptr (int32) as permutation.
    Stable: equal elements are not swapped across phases, preserving original order.
    """
    # We iterate passes = 2*N phases. Even phases update even positions; odd phases update odd positions.
    # For each phase, we only update either even or odd positions.
    for t in range(passes):
        # Determine even or odd phase
        is_even_phase = (t % 2 == 0)
        # For even phase: update positions 0,2,4,...; for odd phase: 1,3,5,...
        # We use a simple per-index update guarded by masks.
        # Launch grid over N; each program handles its own position based on phase.
        # Odd-even sort logic: if is_even_phase:
        #   if i % 2 == 0 and i+1 < N and arr[i] > arr[i+1]: swap both values and indices
        # else:
        #   if i % 2 == 1 and i-1 >= 0 and arr[i] < arr[i-1]: swap both values and indices
        # We implement per-index logic here. Note: Triton supports per-thread control flow.
        for i in range(0, N):
            # even phase: compare i and i+1 when i even
            if is_even_phase:
                even = (i % 2 == 0)
                valid_next = i + 1 < N
                if even and valid_next:
                    a = tl.load(arr_ptr + i)
                    b = tl.load(arr_ptr + (i + 1))
                    # Ascending order: if a > b, swap
                    if a > b:
                        # swap values
                        tl.store(arr_ptr + i, b)
                        tl.store(arr_ptr + (i + 1), a)
                        # swap indices accordingly
                        idx_i = tl.load(indices_ptr + i)
                        idx_j = tl.load(indices_ptr + (i + 1))
                        tl.store(indices_ptr + i, idx_j)
                        tl.store(indices_ptr + (i + 1), idx_i)
            else:
                odd = (i % 2 == 1)
                valid_prev = i - 1 >= 0
                if odd and valid_prev:
                    a = tl.load(arr_ptr + i)
                    prev_a = tl.load(arr_ptr + (i - 1))
                    # Ascending order: if a < prev_a, swap
                    if a < prev_a:
                        tl.store(arr_ptr + i, prev_a)
                        tl.store(arr_ptr + (i - 1), a)
                        idx_i = tl.load(indices_ptr + i)
                        idx_prev = tl.load(indices_ptr + (i - 1))
                        tl.store(indices_ptr + i, idx_prev)
                        tl.store(indices_ptr + (i - 1), idx_i)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is a constant per the original code: 256
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Flatten and prepare data
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Allocate device-side buffers for sort
        # Create arr as a copy of flat (int32)
        arr = flat.to(torch.int32).contiguous()
        # Create indices as 0..N-1 (int32), permutation output
        indices = torch.arange(N, dtype=torch.int32, device=flat.device)

        # 1) Build per-expert counts via Triton histogram
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        # Choose a reasonable block size; Triton will unroll the loop over BLOCK
        BLOCK = 256
        _histogram_counts_kernel[(1,)](arr, counts, N, BLOCK)

        # 2) Compute expert_offsets via Triton inclusive prefix sum
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        # Initialize offsets[0] to 0 (we'll compute the rest in kernel)
        # Run Triton inclusive prefix sum kernel
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # 3) Triton-based stable sort: odd-even transposition sort
        # We perform 2*N passes to guarantee sorting
        _odd_even_sort_stable[(1,)](arr, indices, N, 2 * N)

        # Return sorted_token_indices (indices of permutation) and expert_offsets
        # sorted_token_indices in original is int64; here we return int32 permutation (indices).
        # If exact dtype matching is required, cast to int64, but original returns int32 for indices as well.
        # Also return offsets (int32) as in original's offsets concept. Note: original returns offsets length num_experts+1.
        return indices, offsets


# The original helper functions can remain the same.
def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    return {"topk_idx": topk_idx}


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    """
    Original run helper for reference; not used in evaluator but provided here.
    """
    num_experts = 256
    flat = topk_idx.reshape(-1)
    _, sorted_token_indices = flat.sort(stable=True)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    # Build counts and prefix sum (PyTorch)
    # counts = torch.bincount(flat.long(), minlength=num_experts)
    # expert_offsets[1:] = counts.cumsum(0)
    # Simplified in ModelNew using Triton
    return sorted_token_indices.to(torch.int32), expert_offsets


class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)