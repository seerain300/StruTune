import torch
import triton
import triton.language as tl


# Triton kernel: For each expert key k, count occurrences in 'flat'.
# grid = (NUM_EXPERTS,)
@triton.jit
def counts_kernel(
    flat_ptr,           # *int32, flattened values (length M)
    counts_ptr,         # *int32, output counts per key (length NUM_EXPERTS)
    M,                  # int32, total number of elements
    NUM_EXPERTS: tl.constexpr,  # number of experts (256)
    BLOCK_SIZE: tl.constexpr     # chunk size when scanning flat
):
    k = tl.program_id(0)  # one program per key
    if k >= NUM_EXPERTS:
        return
    # Accumulate count for this key over chunks of flat
    total = tl.zeros((), dtype=tl.int32)
    offset = 0
    while offset < M:
        idxs = offset + tl.arange(0, BLOCK_SIZE)
        mask = idxs < M
        vals = tl.load(flat_ptr + idxs, mask=mask, other=0).to(tl.int32)
        matches = (vals == k) & mask
        total += tl.sum(matches.to(tl.int32), axis=0)
        offset += BLOCK_SIZE
    tl.store(counts_ptr + k, total)


# Triton kernel: Compute inclusive prefix sums offsets_incl[k] = sum_{e=0..k} counts[e].
# grid = (NUM_EXPERTS,)
@triton.jit
def prefix_inclusive_kernel(
    counts_ptr,            # *int32, counts per key (length NUM_EXPERTS)
    offsets_incl_ptr,      # *int32, output inclusive prefix sums (length NUM_EXPERTS)
    NUM_EXPERTS: tl.constexpr
):
    k = tl.program_id(0)  # one program per key
    if k >= NUM_EXPERTS:
        return
    prefix = tl.zeros((), dtype=tl.int32)
    # Loop over keys 0..k and accumulate counts
    t = 0
    while t <= k:
        prefix += tl.load(counts_ptr + t).to(tl.int32)
        t += 1
    tl.store(offsets_incl_ptr + k, prefix)


# Triton kernel: Compute the stable permutation indices sorted_token_indices[j] using key_j = flat[j].
# One program scans all M elements. grid = (1,)
@triton.jit
def stable_permutation_kernel(
    flat_ptr,                 # *int32, flattened values (length M)
    sorted_ptr,               # *int32, output permutation indices (length M)
    offsets_incl_ptr,         # *int32, inclusive prefix sums (length NUM_EXPERTS)
    M,                       # int32, total number of elements
    NUM_EXPERTS: tl.constexpr
):
    j = 0
    while j < M:
        val_j = tl.load(flat_ptr + j).to(tl.int32)
        # base_excl: inclusive prefix sum up to (key-1)
        base_excl = tl.zeros((), dtype=tl.int32)
        if val_j > 0:
            base_excl = tl.load(offsets_incl_ptr + (val_j - 1)).to(tl.int32)

        # tie_count: count of previous elements t < j with same key and flat[t] < flat[j]
        tie_count = tl.zeros((), dtype=tl.int32)
        t = 0
        while t < j:
            val_t = tl.load(flat_ptr + t).to(tl.int32)
            is_same_key = val_t == val_j
            is_smaller_value = val_t < val_j
            tie_count += (is_same_key & is_smaller_value).to(tl.int32)
            t += 1

        rank = base_excl + tie_count
        tl.store(sorted_ptr + j, rank)
        j += 1


# Triton kernel: Write expert_offsets: offsets[e] = offsets_incl[e] for e in [0..NUM_EXPERTS-1],
# and offsets[NUM_EXPERTS] = total_count + 1, where total_count is read from total_count_ptr.
# grid = (1,)
@triton.jit
def finalize_offsets_kernel(
    offsets_incl_ptr,    # *int32, inclusive prefix sums (length NUM_EXPERTS)
    expert_offsets_ptr,  # *int32, output (length NUM_EXPERTS + 1)
    total_count_ptr,     # *int32, scalar total count across all keys
    NUM_EXPERTS: tl.constexpr
):
    e = 0
    while e < NUM_EXPERTS:
        tl.store(expert_offsets_ptr + e, tl.load(offsets_incl_ptr + e).to(tl.int32))
        e += 1
    total = tl.load(total_count_ptr).to(tl.int32)
    tl.store(expert_offsets_ptr + NUM_EXPERTS, total + 1)


# Helper Triton kernel to reduce sum of counts into total_count (used to finalize offsets).
# grid = (1,)
@triton.jit
def reduce_sum_kernel(counts_ptr, total_ptr, N: tl.constexpr):
    acc = tl.zeros((), dtype=tl.int32)
    i = 0
    while i < N:
        acc += tl.load(counts_ptr + i).to(tl.int32)
        i += 1
    tl.store(total_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int, block_size: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.block_size = block_size

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure on CUDA
        assert topk_idx.is_cuda, "ModelNew requires CUDA tensors."
        # Flatten and make contiguous
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        NUM_EXPERTS = self.num_experts

        # 1) Triton: counts_kernel (grid=(NUM_EXPERTS,))
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        counts_kernel[(NUM_EXPERTS,)](
            flat, counts, M, NUM_EXPERTS, self.block_size
        )

        # 2) Triton: prefix_inclusive_kernel (grid=(NUM_EXPERTS,))
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        prefix_inclusive_kernel[(NUM_EXPERTS,)](
            counts, offsets_incl, NUM_EXPERTS
        )

        # 3) Triton: compute total_count = sum(counts) (grid=(1,))
        total_count = torch.empty(1, dtype=torch.int32, device=flat.device)
        reduce_sum_kernel[(1,)](counts, total_count, NUM_EXPERTS)

        # 4) Triton: finalize_offsets_kernel (grid=(1,))
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        finalize_offsets_kernel[(1,)](
            offsets_incl, expert_offsets, total_count, NUM_EXPERTS
        )

        # 5) Triton: stable_permutation_kernel (grid=(1,)) computes sorted_token_indices
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)
        stable_permutation_kernel[(1,)](
            flat, sorted_token_indices, offsets_incl, M, NUM_EXPERTS
        )

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
