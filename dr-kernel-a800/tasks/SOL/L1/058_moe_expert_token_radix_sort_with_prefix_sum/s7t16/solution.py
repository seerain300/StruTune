import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements; atomically increments counts[value].
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        if mask[i]:
            v = vals[i]
            if (v >= 0) & (v < NUM_VALUES):
                tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    # Compute inclusive prefix sums: prefix[i] = sum_{x=0..i} counts[x].
    # Do in a single program; NUM_VALUES is small (256).
    for i in range(NUM_VALUES):
        total = 0
        for j in range(i + 1):
            total += tl.load(counts_ptr + j)
        tl.store(prefix_ptr + i, total)


@triton.jit
def assemble_offsets(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    # offsets[0] = 0; offsets[i+1] = prefix[i] for i in [0..NUM_VALUES-1].
    tl.store(offsets_ptr + 0, 0)
    for i in range(NUM_VALUES):
        tl.store(offsets_ptr + 1 + i, tl.load(prefix_ptr + i))


@triton.jit
def stable_permutation_kernel(flat_ptr, sorted_idx_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # Build stable permutation indices: sorted_idx[i] = position of i in sorted flat.
    # We perform per-value scans to ensure stability for equal values using original positions as tie-breaker.
    for v in range(NUM_VALUES):
        # First, compute number_of_less = count of elements strictly less than v.
        number_of_less = 0
        base = 0
        while base < M:
            offsets = base + tl.arange(0, BLOCK)
            mask = offsets < M
            vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
            for i in range(BLOCK):
                if mask[i]:
                    if vals[i] < v:
                        number_of_less += 1
            base += BLOCK

        # Next, write positions for all i where flat[i] == v, using original positions as tie-breaker.
        # Maintain a running count of how many elements equal to v have already been placed (eq_count).
        # Each i with vi == v is placed at position number_of_less + eq_count, then eq_count++.
        eq_count = 0
        base = 0
        while base < M:
            offsets = base + tl.arange(0, BLOCK)
            mask = offsets < M
            vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
            pos_vec = offsets.to(tl.int32)  # original position i
            for i in range(BLOCK):
                if mask[i]:
                    vi = vals[i]
                    if vi == v:
                        tl.store(sorted_idx_ptr + (number_of_less + eq_count), pos_vec[i])
                        eq_count += 1
            base += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts  # In the provided workloads, this is 256.

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32 tensor on device.
        Returns:
          sorted_token_indices: (M,) int32 tensor, positions that would sort the flattened indices stably.
          expert_offsets: (num_experts+1,) int32 tensor, inclusive prefix of counts per value.
        """
        device = topk_idx.device
        # Flatten to 1D int32
        original_flat = topk_idx.reshape(-1).to(torch.int32)
        M = original_flat.numel()
        NUM_VALUES = self.num_experts  # 256 in the provided workloads

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)

        # 1) Histogram of values in [0..NUM_VALUES-1]
        BLOCK = 1024
        grid_hist = (triton.cdiv(M, BLOCK),)
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        histogram_kernel[grid_hist](original_flat, counts, M, NUM_VALUES, BLOCK)

        # 2) Inclusive prefix sums
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES)

        # 3) Assemble offsets
        assemble_offsets[(1,)](prefix, offsets, NUM_VALUES)

        # 4) Stable permutation via Triton (no torch.sort)
        grid_perm = (triton.cdiv(M, BLOCK),)
        stable_permutation_kernel[grid_perm](original_flat, sorted_token_indices, M, NUM_VALUES, BLOCK)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
