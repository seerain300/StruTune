import torch
import triton
import triton.language as tl


# Triton kernel: histogram of flattened int32 values into counts[0..NUM_VALUES-1].
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # For each valid element, atomically increment its count
    for i in range(BLOCK):
        if mask[i]:
            v = vals[i].to(tl.int32)
            # Guard to ensure v in [0, NUM_VALUES-1] (not strictly necessary if inputs are generated correctly)
            if v >= 0 and v < NUM_VALUES:
                tl.atomic_add(counts_ptr + v, 1)


# Triton kernel: inclusive prefix sum of counts across NUM_VALUES.
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    # Single-program loop over NUM_VALUES
    for i in range(NUM_VALUES):
        sum_i = 0
        # Sum counts[0..i] sequentially
        for j in range(0, i + 1):
            sum_i += tl.load(counts_ptr + j)
        tl.store(prefix_ptr + i, sum_i)


# Triton kernel: assemble offsets as prefix[0..NUM_VALUES-1] + 0 at start
@triton.jit
def assemble_offsets(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    # offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # offsets[i+1] = prefix[i] for i in [0..NUM_VALUES-1]
    for i in range(NUM_VALUES):
        tl.store(offsets_ptr + 1 + i, tl.load(prefix_ptr + i))


# Triton kernel: stable permutation for 1D int32 array with values in [0..NUM_VALUES-1].
# Produces sorted_token_indices that would sort original_flat stably.
@triton.jit
def stable_permutation_kernel(original_ptr, sorted_indices_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # For each value v, compute number_of_less and then place equal elements in stable order using original positions.
    for v in range(NUM_VALUES):
        number_of_less = 0
        # count elements strictly less than v
        base = 0
        while base < M:
            offsets = base + tl.arange(0, BLOCK)
            mask = offsets < M
            vals = tl.load(original_ptr + offsets, mask=mask, other=0).to(tl.int32)
            for i in range(BLOCK):
                if mask[i]:
                    if vals[i] < v:
                        number_of_less += 1
            base += BLOCK

        # Now place all elements equal to v at positions: base_pos = number_of_less + number_of_equal_before_i
        base = 0
        while base < M:
            offsets = base + tl.arange(0, BLOCK)
            mask = offsets < M
            vals = tl.load(original_ptr + offsets, mask=mask, other=0).to(tl.int32)
            for i in range(BLOCK):
                if mask[i]:
                    if vals[i] == v:
                        # number_of_equal_before_i: count how many equal elements appear before i in original order
                        eq_before = 0
                        j = 0
                        while j < i:
                            if mask[j]:
                                if vals[j] == v:
                                    eq_before += 1
                            j += 1
                        pos = number_of_less + eq_before
                        # Store index i at position pos in sorted_indices
                        tl.store(sorted_indices_ptr + pos, offsets[i].to(tl.int32))
            base += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure device and dtype
        device = topk_idx.device
        original_flat = topk_idx.reshape(-1).contiguous()  # int32, 1D

        M = original_flat.numel()
        NUM_VALUES = 256  # in provided workloads, num_experts_per_tok = 256
        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)

        # 1) Histogram
        BLOCK = 1024
        grid_hist = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid_hist](original_flat, counts, M, NUM_VALUES, BLOCK)

        # 2) Inclusive prefix sum of counts
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES)

        # 3) Assemble offsets
        assemble_offsets[(1,)](prefix, offsets, NUM_VALUES)

        # 4) Stable permutation using Triton (no torch.sort)
        stable_permutation_kernel[grid_hist](original_flat, sorted_token_indices, M, NUM_VALUES, BLOCK)

        # Return results matching original run: (sorted_token_indices, expert_offsets)
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
