import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(original_ptr + offsets, mask=mask, other=-1)
    # Atomic add 1 for each valid position equal to v, for all v in [0..NUM_VALUES-1]
    for v in range(NUM_VALUES):
        is_eq = vals == v
        # Convert boolean mask to int32 and sum valid positions
        add_mask = is_eq & mask
        add_val = tl.sum(add_mask.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + v, add_val)


@triton.jit
def prefix_sum_inclusive(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    # Inclusive prefix sum: prefix[i] = sum_{j<=i} counts[j]
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(NUM_VALUES):
        val = tl.load(counts_ptr + i)
        acc += val
        tl.store(prefix_ptr + i, acc)


@triton.jit
def assemble_offsets(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    # offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # offsets[i+1] = prefix[i] for i in 0..NUM_VALUES-1
    for i in range(NUM_VALUES):
        acc = tl.load(prefix_ptr + i)
        tl.store(offsets_ptr + 1 + i, acc)


@triton.jit
def stable_permutation_kernel(original_ptr, sorted_ptr, M, offsets_ptr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M

    for v in range(NUM_VALUES):
        # Compute number_of_less = sum_{u<v} counts[u] using prefix[v-1] if v>0 else 0
        number_of_less = tl.zeros((), dtype=tl.int32)
        if v > 0:
            number_of_less = tl.load(offsets_ptr + v - 1)

        # For each element in this tile, place if value == v
        group = tl.zeros((), dtype=tl.int32)
        for i in range(BLOCK):
            idx = offsets[i]
            m = mask[i]
            val = tl.load(original_ptr + idx, mask=m, other=-1)
            if val == v:
                pos = number_of_less + group
                tl.store(sorted_ptr + pos, idx)
                group += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure int32 and flatten
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        original_flat = topk_idx.contiguous().view(-1).to(torch.int32)
        M = original_flat.numel()
        NUM_VALUES = 256  # consistent with the workload
        device = original_flat.device

        # Outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        expert_offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)

        # 1) Triton histogram counts of values [0..255]
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_h = triton.cdiv(M, BLOCK_HIST)
        histogram_kernel[grid_h](original_flat, counts, M, NUM_VALUES, BLOCK_HIST)

        # 2) Triton inclusive prefix sum of counts -> prefix[i] = sum_{j<=i} counts[j]
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        prefix_sum_inclusive[(1,)](counts, prefix, NUM_VALUES)

        # 3) Assemble expert_offsets from prefix
        assemble_offsets[(1,)](prefix, expert_offsets, NUM_VALUES)

        # 4) Triton stable permutation to produce sorted_token_indices
        BLOCK_PERM = 1024
        grid_p = triton.cdiv(M, BLOCK_PERM)
        stable_permutation_kernel[grid_p](original_flat, sorted_token_indices, M, expert_offsets, NUM_VALUES, BLOCK_PERM)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
