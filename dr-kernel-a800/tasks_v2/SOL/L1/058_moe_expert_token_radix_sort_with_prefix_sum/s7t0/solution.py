import torch
import triton
import triton.language as tl


# Triton kernel: compute histogram of flattened indices (int32).
# For each element in flat, atomically increment counts[v].
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M, BLOCK: tl.constexpr, NUM_EXPERTS: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        v = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + v, 1)


# Triton kernel: compute inclusive prefix sum of counts -> prefix[i] = sum_{j<=i} counts[j]
# Writes to prefix_ptr[i] for i in [0..NUM_EXPERTS-1].
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_EXPERTS: tl.constexpr):
    prefix = tl.zeros((), dtype=tl.int32)
    tl.atomic_add(prefix_ptr + 0, 0)  # prefix[0] = 0
    i = 0
    while i < NUM_EXPERTS:
        count_i_minus_1 = tl.load(counts_ptr + i)
        tl.atomic_add(prefix_ptr + (i + 1), prefix)
        prefix += count_i_minus_1
        i += 1


# Triton kernel: fill sorted_token_indices with counting sort using counts and prefix.
# Launch with grid=(NUM_EXPERTS,), each program handles one expert e.
# Each program writes exactly 'c' elements starting at write_index = prefix[e].
@triton.jit
def counting_sort_write_kernel(counts_ptr, prefix_ptr, sorted_ptr, NUM_EXPERTS: tl.constexpr):
    e = tl.program_id(0)  # expert index
    # Load count for this expert
    c = tl.load(counts_ptr + e)
    # If c == 0, nothing to write
    if c == 0:
        return
    # Compute starting write index: prefix[e] (inclusive count of tokens for experts < e)
    start = tl.load(prefix_ptr + e)
    # Write e at positions start + 0 .. start + c - 1
    # We'll do this in a small loop; since c <= total tokens and NUM_EXPERTS is small, this is fine.
    j = 0
    while j < c:
        # sorted_ptr is int32; we store e (int32)
        tl.store(sorted_ptr + (start + j), e)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - sorted_token_indices: produced by Triton counting sort (no torch.sort).
        - expert_offsets: produced by Triton histogram and prefix sum.
        """
        if not topk_idx.is_cuda:
            raise RuntimeError("ModelNew expects topk_idx on a CUDA device.")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.view(-1)  # int32
        M = flat.numel()

        num_experts = 256  # match original

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # 1) Histogram: one atomic add per element
        BLOCK = 1024
        grid = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid](flat, counts, M, BLOCK, NUM_EXPERTS=num_experts)

        # 2) Prefix sum of counts: inclusive prefix
        prefix = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        prefix_sum_kernel[(1,)](counts, prefix, NUM_EXPERTS=num_experts)

        # 3) Counting sort write: for each expert e, write its tokens at position prefix[e] + j
        # Launch one program per expert
        counting_sort_write_kernel[(num_experts,)](counts, prefix, sorted_token_indices, NUM_EXPERTS=num_experts)

        # 4) expert_offsets: we already have prefix as inclusive prefix sums up to each expert.
        #   offsets[i] = prefix[i] for i in [0..NUM_EXPERTS]. We need offsets of length NUM_EXPERTS + 1.
        #   offsets[0] = 0, offsets[i+1] = prefix[i] for i in [0..NUM_EXPERTS-1].
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0
        for i in range(num_experts):
            expert_offsets[i + 1] = prefix[i]

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
