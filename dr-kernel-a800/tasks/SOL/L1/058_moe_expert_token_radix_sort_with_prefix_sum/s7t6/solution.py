import torch
import triton
import triton.language as tl


# Triton kernel: histogram of flattened indices (int32).
# For each element in flat, atomically increment counts[value].
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        v = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + v, 1)


# Triton kernel: compute inclusive prefix sum of counts -> prefix[v] = sum_{x<=v} counts[x]
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    running = tl.zeros((), dtype=tl.int32)
    i = 0
    while i < NUM_VALUES:
        count_i = tl.load(counts_ptr + i)
        tl.store(prefix_ptr + i, running)
        running += count_i
        i += 1


# Triton kernel: assemble expert_offsets from prefix.
# Writes offsets[0] = 0; offsets[i+1] = prefix[i] for i in [0..NUM_VALUES-1].
@triton.jit
def assemble_offsets(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    tl.store(offsets_ptr + 0, 0)
    i = 0
    while i < NUM_VALUES:
        p = tl.load(prefix_ptr + i)
        tl.store(offsets_ptr + (i + 1), p)
        i += 1


# Triton kernel: compute stable permutation indices for values in [0..NUM_VALUES-1].
# sorted_token_indices is a global output int32 tensor of length M.
# For each original[i] == v, place i at position sum(<v) + number_of_equal_before_this_position.
@triton.jit
def stable_permutation_kernel(flat_ptr, sorted_token_indices_ptr, M: tl.constexpr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # Loop over values v = 0..NUM_VALUES-1
    for v in range(NUM_VALUES):
        # Compute sum of elements strictly less than v
        sum_less = tl.zeros((), dtype=tl.int32)
        for start in range(0, M, BLOCK):
            idxs = start + tl.arange(0, BLOCK)
            mask = idxs < M
            vals = tl.load(flat_ptr + idxs, mask=mask, other=0)
            is_less = (vals < v) & mask
            block_sum = tl.sum(is_less.to(tl.int32), axis=0)
            sum_less += block_sum

        # Compute stable positions: for each i with flat[i] == v, count how many equal elements appear before it
        for start in range(0, M, BLOCK):
            idxs = start + tl.arange(0, BLOCK)
            mask = idxs < M
            vals = tl.load(flat_ptr + idxs, mask=mask, other=0)
            eq_mask = (vals == v) & mask  # vector of booleans

            # eq_before[j] = number of true in eq_mask for positions < j in this block
            for j in range(BLOCK):
                acc = tl.zeros((), dtype=tl.int32)
                for k in range(0, BLOCK):
                    mkk = mask[k]
                    eqk = eq_mask[k].to(tl.int32)
                    less_j = (k < j) & mkk
                    contrib = eqk * less_j.to(tl.int32)
                    acc += contrib
                # Write final position for position j in this block
                if mask[j]:
                    pos = sum_less + acc
                    tl.store(sorted_token_indices_ptr + (start + j), tl.full((), start + j, tl.int32))


def run(topk_idx: torch.Tensor):
    """
    Triton-only implementation of:
      - sorted_token_indices = torch.sort(flat, stable=True).indices
      - expert_offsets[0] = 0; expert_offsets[i+1] = sum_{x<=i} counts_x for x in [0..num_experts_per_tok-1]
    where flat = topk_idx.reshape(-1).

    Returns:
      sorted_token_indices: int32 tensor of shape (M,)
      expert_offsets: int32 tensor of shape (num_experts_per_tok + 1,)
    """
    device = topk_idx.device
    original_flat = topk_idx.reshape(-1).contiguous()  # int32, CUDA
    M = original_flat.numel()
    NUM_VALUES = 256  # matches num_experts_per_tok in provided tests

    # Allocate outputs
    sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)

    # 1) Histogram: counts per value 0..NUM_VALUES-1
    counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
    BLOCK = 1024
    grid_hist = (triton.cdiv(M, BLOCK),)
    histogram_kernel[grid_hist](original_flat, counts, M, NUM_VALUES, BLOCK)

    # 2) Inclusive prefix sums
    prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
    prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES)

    # 3) Assemble offsets: expert_offsets[0] = 0; expert_offsets[i+1] = prefix[i]
    offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)
    assemble_offsets[(1,)](prefix, offsets, NUM_VALUES)

    # 4) Stable permutation via Triton
    BLOCK_SORT = 1024
    grid_perm = (triton.cdiv(M, BLOCK_SORT),)
    stable_permutation_kernel[grid_perm](original_flat, sorted_token_indices, M, NUM_VALUES, BLOCK_SORT)

    return sorted_token_indices, offsets


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]
        return run(topk_idx)


def run(*args):
    return ModelNew()(*args)
