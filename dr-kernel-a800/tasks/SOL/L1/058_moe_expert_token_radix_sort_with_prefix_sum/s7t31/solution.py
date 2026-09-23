import torch
import triton
import triton.language as tl


# Kernel: histogram of flattened int32 values into counts[0..255]
@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    # Load int32 values
    values = tl.load(original_ptr + offsets, mask=mask, other=0)
    # Atomic add 1 for each valid value; values are in [0..255]
    tl.atomic_add(counts_ptr + values, 1, mask=mask)


# Kernel: exclusive prefix sum of counts to produce number_of_less per value.
# Iterates over bins v=0..254; updates prefix[v+1] += prefix[v].
# Small loop, launched with a single program.
@triton.jit
def exclusive_cumsum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    for v in range(NUM_VALUES - 1):
        prev = tl.load(prefix_ptr + v)
        curr = tl.load(prefix_ptr + v + 1)
        tl.store(prefix_ptr + v + 1, curr + prev)


# Kernel: assemble offsets: offsets[0]=0; offsets[i+1]=offsets[i] + counts[i]
@triton.jit
def assemble_offsets_kernel(counts_ptr, offsets_ptr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # Single program handles assembly
    if tl.program_id(0) == 0:
        tl.store(offsets_ptr + 0, 0)
        for v in range(NUM_VALUES):
            tl.store(offsets_ptr + v + 1, tl.load(offsets_ptr + v) + tl.load(counts_ptr + v))
    else:
        # No-op for other pids
        pass


# Kernel: stable permutation by counting-sort logic:
# For each value v in [0..255]:
#   number_of_less = sum_{x<v} counts[x]
#   For i in 0..M-1: if original[i] == v, store i at position 'pos' where:
#       pos = number_of_less + (number of elements equal to v encountered before i).
# Because there are no duplicates across elements, pos is unique and stable.
@triton.jit
def stable_permutation_kernel(original_ptr, sorted_ptr, counts_ptr, prefix_ptr, M: tl.int32, NUM_VALUES: tl.constexpr):
    # Single program handles entire permutation
    for v in range(NUM_VALUES):
        number_of_less = tl.load(prefix_ptr + v)
        # Track how many elements equal to 'v' have been processed before this i.
        # For each element, if original[i] == v, write i at position 'pos' where pos = number_of_less + eq_seen.
        for i in range(M):
            val = tl.load(original_ptr + i)
            if val == v:
                # For this value, no duplicates, so eq_seen increments deterministically.
                # We need to know how many 'v' elements were processed before i.
                # Compute eq_seen as the number of j < i with original[j] == v.
                eq_seen = 0
                # Iterate j from 0 to i-1; for v in [0..255] and M up to a few thousand, this is acceptable.
                for j in range(i):
                    vj = tl.load(original_ptr + j)
                    if vj == v:
                        eq_seen += 1
                pos = number_of_less + eq_seen
                tl.store(sorted_ptr + pos, i)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure int32 and contiguous
        original = topk_idx.to(torch.int32).contiguous()
        M = original.numel()
        device = original.device

        # Outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        # prefix for number_of_less per value
        prefix = torch.empty(256, dtype=torch.int32, device=device)

        # 1) Histogram of values
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(M, BLOCK_HIST),)
        histogram_kernel[grid_hist](original, counts, M, BLOCK_HIST)

        # 2) Exclusive prefix sum to get number_of_less per value
        exclusive_cumsum_kernel[(1,)](counts, prefix, 256)

        # 3) Assemble offsets: offsets[0]=0; offsets[i+1]=offsets[i] + counts[i]
        BLOCK_OFFSET = 1  # single program handles assembly
        assemble_offsets_kernel[(BLOCK_OFFSET,)](counts, offsets, 256, 1)

        # 4) Stable permutation of original_flat to sorted_token_indices
        stable_permutation_kernel[(1,)](original, sorted_token_indices, counts, prefix, M, 256)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
