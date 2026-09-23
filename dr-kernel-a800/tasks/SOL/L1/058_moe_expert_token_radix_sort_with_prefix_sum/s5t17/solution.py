import torch
import triton
import triton.language as tl


# Triton kernel 1: For each expert key k, count occurrences in 'flat' and compute offsets_incl[k] = sum_{e=0..k} counts[e].
@triton.jit
def counts_and_bases_kernel(
    flat_ptr,                  # *int32, flattened values (length M)
    counts_ptr,                # *int32, output counts per key (length NUM_EXPERTS)
    offsets_incl_ptr,          # *int32, output inclusive prefix sums (length NUM_EXPERTS)
    total_count_ptr,           # *int32, scalar total count across all keys
    M: tl.constexpr,           # total number of elements (compile-time for loop)
    NUM_EXPERTS: tl.constexpr, # number of experts (e.g., 256)
    BLOCK_SIZE: tl.constexpr    # chunk size when scanning flat
):
    k = tl.program_id(0)  # one program per key in [0..NUM_EXPERTS)
    if k >= NUM_EXPERTS:
        return

    # Accumulate count for key k across flat
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

    # Compute inclusive prefix sum for this key: offsets_incl[k] = sum_{e=0..k} counts[e]
    prefix = tl.zeros((), dtype=tl.int32)
    t = 0
    while t <= k:
        cnt = tl.load(counts_ptr + t).to(tl.int32)
        prefix += cnt
        t += 1
    tl.store(offsets_incl_ptr + k, prefix)

    # Store total count (sum of all counts) for later use
    tl.atomic_add(total_count_ptr, total)


# Triton kernel 2: Finalize expert_offsets from offsets_incl and total_count.
@triton.jit
def finalize_offsets_kernel(
    offsets_incl_ptr,     # *int32, length NUM_EXPERTS, inclusive prefix sums of counts
    expert_offsets_ptr,   # *int32, output length NUM_EXPERTS+1
    total_count_ptr,      # *int32, scalar total count
    NUM_EXPERTS: tl.constexpr
):
    # Fill expert_offsets[e] = offsets_incl[e] for e in [0..NUM_EXPERTS-1]
    for e in range(NUM_EXPERTS):
        tl.store(expert_offsets_ptr + e, tl.load(offsets_incl_ptr + e))
    # Fill expert_offsets[NUM_EXPERTS] = total_count + 1
    total = tl.load(total_count_ptr).to(tl.int32)
    tl.store(expert_offsets_ptr + NUM_EXPERTS, total + 1)


# Triton kernel 3: Compute stable argsort permutation indices by ranking.
@triton.jit
def stable_permutation_kernel(
    flat_ptr,             # *int32, flattened values (length M)
    sorted_idx_ptr,       # *int32, output permutation indices (length M)
    offsets_incl_ptr,     # *int32, inclusive prefix sums per key (length NUM_EXPERTS)
    M: tl.constexpr,      # total number of elements
    NUM_EXPERTS: tl.constexpr
):
    # For each linear index j in [0..M), compute its sorted position using stable ranking:
    # rank_j = base_excl + tie_count, where
    #   base_excl = offsets_incl[key_j - 1] if key_j > 0 else 0
    #   tie_count = number of t < j with same key and flat[t] < flat[j]
    for j in range(M):
        val_j = tl.load(flat_ptr + j).to(tl.int32)
        # base excl
        if val_j == 0:
            base_excl = tl.zeros((), dtype=tl.int32)
        else:
            base_excl = tl.load(offsets_incl_ptr + (val_j - 1))
        # tie_count = count of earlier t with key==val_j and flat[t] < val_j
        tie_count = tl.zeros((), dtype=tl.int32)
        for t in range(j):
            val_t = tl.load(flat_ptr + t).to(tl.int32)
            if (val_t == val_j) and (val_t < val_j):
                tie_count += 1
        rank = base_excl + tie_count
        tl.store(sorted_idx_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block_size: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.block_size = block_size

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure on CUDA
        assert topk_idx.is_cuda, "ModelNew requires a CUDA tensor."
        # Flatten and make contiguous
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        NUM_EXPERTS = self.num_experts

        # 1) Triton: counts_and_bases_kernel (grid=(NUM_EXPERTS,))
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        total_count_buf = torch.zeros(1, dtype=torch.int32, device=flat.device)

        counts_and_bases_kernel[(NUM_EXPERTS,)](
            flat, counts, offsets_incl, total_count_buf, M, NUM_EXPERTS, self.block_size
        )

        # 2) Triton: finalize_offsets_kernel (grid=(1,))
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        finalize_offsets_kernel[(1,)](
            offsets_incl, expert_offsets, total_count_buf, NUM_EXPERTS
        )

        # 3) Triton: stable permutation kernel (grid=(1,))
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)
        stable_permutation_kernel[(1,)](
            flat, sorted_token_indices, offsets_incl, M, NUM_EXPERTS
        )

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
