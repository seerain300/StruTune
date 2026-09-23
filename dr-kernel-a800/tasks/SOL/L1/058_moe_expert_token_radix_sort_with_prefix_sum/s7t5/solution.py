import torch
import triton
import triton.language as tl


# Triton kernel: histogram of flattened indices (int32).
# For each element in flat, atomically increment counts[value % NUM_VALUES].
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M: tl.constexpr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        v = vals[i]
        if mask[i]:
            idx = v % NUM_VALUES
            tl.atomic_add(counts_ptr + idx, 1)


# Triton kernel: compute inclusive prefix sum of counts -> prefix[v] = sum_{x<=v} counts[x]
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    prefix = tl.zeros((), dtype=tl.int32)
    tl.atomic_add(prefix_ptr + 0, 0)  # prefix[0] = 0
    i = 0
    while i < NUM_VALUES:
        count_i = tl.load(counts_ptr + i)
        tl.atomic_add(prefix_ptr + (i + 1), prefix)
        prefix += count_i
        i += 1


# Triton kernel: assemble expert_offsets from prefix:
# offsets[0] = 0; offsets[i+1] = prefix[i] for i in [0..NUM_VALUES-1].
@triton.jit
def assemble_offsets(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    tl.store(offsets_ptr + 0, 0)
    i = 0
    while i < NUM_VALUES:
        p = tl.load(prefix_ptr + i)
        tl.store(offsets_ptr + (i + 1), p)
        i += 1


# Triton kernel: generate stable permutation sorted_token_indices based on prefix.
# For each original element at index i: read value v = original[i],
# compute position pos = prefix[v], write i into sorted_idx[pos].
# NUM_VALUES must match the valid range of original values (e.g., 256).
@triton.jit
def stable_permutation_kernel(original_ptr, prefix_ptr, sorted_idx_ptr, M: tl.constexpr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    # For each i, determine v = original[i], pos = prefix[v], then set sorted_idx[pos] = i
    for i in range(BLOCK):
        if mask[i]:
            v = tl.load(original_ptr + offsets[i])  # v in [0..NUM_VALUES-1]
            pos = tl.load(prefix_ptr + v)          # position in sorted order for value v
            tl.store(sorted_idx_ptr + pos, offsets[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch_size, seq_len, num_experts_per_tok), int32, CUDA
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        M = topk_idx.numel()
        device = topk_idx.device
        original_flat = topk_idx.reshape(-1).to(torch.int32)

        # Valid expert range (per original code behavior: indices in [0, num_experts-1], which equals num_experts_per_tok=256)
        NUM_VALUES = topk_idx.shape[2]

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)

        # 1) Histogram of values in [0..NUM_VALUES-1]
        BLOCK = 1024
        grid_hist = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid_hist](original_flat, counts, M, NUM_VALUES, BLOCK)

        # 2) Inclusive prefix sums
        prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES)

        # 3) Assemble offsets
        assemble_offsets[(1,)](prefix, offsets, NUM_VALUES)

        # 4) Stable permutation: sorted_token_indices = argsort positions based on prefix
        #    For each original[i] == v, place i at position prefix[v].
        grid_perm = (triton.cdiv(M, BLOCK),)
        stable_permutation_kernel[grid_perm](original_flat, prefix, sorted_token_indices, M, NUM_VALUES, BLOCK)

        # Return: sorted_token_indices (int32, length M), and expert_offsets (int32, length NUM_VALUES+1)
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
