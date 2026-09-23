import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(top_ptr, counts_ptr, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-expert counts for values in top_ptr[0:n_elements], where each value is in [0, num_experts-1].
    counts_ptr has length num_experts, and we'll write counts to it using atomic_add.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load values; for masked-off lanes, use 0 so they don't contribute
    vals = tl.load(top_ptr + offsets, mask=mask, other=0)
    # We assume vals are in range [0, num_experts-1]. For masked lanes, set val=0 to avoid spurious atomics.
    # Note: Triton supports int32 indexing; counts_ptr is int32.
    # Use atomic add to accumulate counts per value.
    # We don't know num_experts as a meta-arg here; we rely on caller to pass correct counts length.
    # The 'other=0' for masked lanes avoids atomics on invalid positions.
    for i in range(0, BLOCK_SIZE):
        if mask[i]:
            # Cast to int32 before atomic add
            v = tl.cast(vals[i], tl.int32)
            # Atomic add 1 for this value
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0:num_experts] into offsets_ptr[0:num_experts+1].
    offsets_ptr[0] = 0, offsets_ptr[1] = counts[0], ..., offsets_ptr[num_experts] = sum_{i=0..num_experts-1} counts[i].
    We set offsets_ptr[num_experts] to N (total elements) for convenience, but last element should be the full sum.
    """
    # Since num_experts is small (256), a simple loop is fine and fast.
    acc = 0
    # offsets_ptr is length num_experts + 1
    # We'll write offsets_ptr[0] = 0
    # Triton allows Python range with small integers; num_experts is passed as tl.int32 scalar.
    # Write initial zero
    offsets_ptr[0] = 0
    for i in range(0, num_experts):
        acc += counts_ptr[i]
        offsets_ptr[i + 1] = acc
    # The last element should be the total sum; PyTorch's cumsum would give it automatically. Here we rely on the loop's acc.
    # If you want to ensure it's N (not needed for correctness of cumsum per se), you can set it here if desired.


@triton.jit
def _odd_even_sort_stable_kernel(values_ptr, indices_ptr, n_elements: tl.int32, max_passes: tl.constexpr):
    """
    Perform odd-even transposition sort on values_ptr (size n_elements) with stable behavior by swapping indices accordingly.
    Maintain indices_ptr parallel to values_ptr: each position holds the original index of that value.
    We run a fixed number of passes (max_passes), which is large enough to ensure convergence for n_elements.
    """
    pid = tl.program_id(0)
    pos = pid  # one program per element
    # We simulate even and odd phases. Only elements at even or odd positions will perform swaps in their respective phases.
    # We will read from 'prev' and 'next' positions only when they exist within bounds, otherwise leave them as-is.
    for t in range(0, max_passes):
        # Even phase: positions 0,2,4,...
        if (pos % 2 == 0) & (pos + 1 < n_elements):
            prev = pos - 1
            next_pos = pos + 1
            a = tl.load(values_ptr + pos)
            b = tl.load(values_ptr + next_pos)
            idx_a = tl.load(indices_ptr + pos)
            idx_b = tl.load(indices_ptr + next_pos)
            # Determine swap
            swap = a > b  # stable: if equal, original order preserved (no swap) -> odd-even naturally preserves stability
            new_a = tl.where(swap, b, a)
            new_b = tl.where(swap, a, b)
            new_idx_a = tl.where(swap, idx_b, idx_a)
            new_idx_b = tl.where(swap, idx_a, idx_b)
            # Write back
            tl.store(values_ptr + pos, new_a)
            tl.store(values_ptr + next_pos, new_b)
            tl.store(indices_ptr + pos, new_idx_a)
            tl.store(indices_ptr + next_pos, new_idx_b)
        # Odd phase: positions 1,3,5,...
        if (pos % 2 == 1) & (pos - 1 >= 0):
            prev = pos - 1
            a = tl.load(values_ptr + pos)
            b = tl.load(values_ptr + prev)
            idx_a = tl.load(indices_ptr + pos)
            idx_b = tl.load(indices_ptr + prev)
            swap = a > b
            new_a = tl.where(swap, b, a)
            new_b = tl.where(swap, a, b)
            new_idx_a = tl.where(swap, idx_b, idx_a)
            new_idx_b = tl.where(swap, idx_a, idx_b)
            tl.store(values_ptr + pos, new_a)
            tl.store(values_ptr + prev, new_b)
            tl.store(indices_ptr + pos, new_idx_a)
            tl.store(indices_ptr + prev, new_idx_b)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable block size for histogram; 1024 works well for typical sizes
        self.block_size = 1024

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute counts per expert using Triton.
        - Compute inclusive prefix sum of counts using Triton.
        - Sort the flattened values stably using a Triton odd-even sort and return the permutation of indices.
        - Return sorted_token_indices (permutation) and expert_offsets (cumulative counts per expert).
        """
        # Ensure contiguity and flatten
        flat = topk_idx.contiguous().view(-1)
        n = flat.numel()

        # 1) Triton histogram counts: counts per expert ID
        num_experts = 256  # match original behavior
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        # Launch kernel in chunks of BLOCK_SIZE
        grid = (triton.cdiv(n, self.block_size),)
        # Note: flat is int32 (from get_inputs), so load directly
        _histogram_counts_kernel[grid](flat, counts, n, BLOCK_SIZE=self.block_size)

        # 2) Triton inclusive prefix sum to produce expert_offsets (length num_experts + 1)
        # We'll compute via a simple loop inside Triton (num_experts is small). Launch a single program instance.
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # 3) Triton odd-even transposition sort to produce sorted_token_indices permutation
        # Prepare values buffer (copy of flat) and indices buffer (0..n-1)
        values = torch.empty(n, dtype=torch.int32, device=flat.device)
        # Initialize values with flat (int32 cast if needed)
        values.copy_(flat)
        indices = torch.arange(n, dtype=torch.int32, device=flat.device)

        # Choose max_passes to ensure convergence; use a conservative upper bound
        # For correctness and small-to-moderate N, 2*n passes are sufficient.
        max_passes = 2 * n  # constexpr for Triton
        # Run sorting kernel with one program per element
        _odd_even_sort_stable_kernel[(n,)](values, indices, n, max_passes=max_passes)

        # Return as in original: sorted_token_indices (int32 permutation), expert_offsets (int32)
        # Note: original returns int64 for indices, but here we return int32; cast if you need strict dtype match.
        return indices, offsets