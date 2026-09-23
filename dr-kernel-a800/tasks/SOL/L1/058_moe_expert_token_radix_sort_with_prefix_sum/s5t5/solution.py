import torch
import triton
import triton.language as tl


# Kernel 1: compute per-expert counts. One program per key k in [0, NUM_EXPERTS).
@triton.jit
def _count_experts_kernel(flat_ptr, counts_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(axis=0)  # key index
    acc = tl.zeros((), dtype=tl.int32)
    # iterate over flat in chunks of BLOCK, masked by j < M
    for j0 in range(0, M, BLOCK):
        idx = j0 + tl.arange(0, BLOCK)
        mask = idx < M
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)  # other=0 to ignore OOB
        # count how many equal to k in this chunk
        cnt_chunk = tl.sum((vals == k).to(tl.int32), axis=0)
        acc += cnt_chunk
    tl.store(counts_ptr + k, acc)


# Kernel 2: inclusive prefix sum of counts -> offsets_incl[e] = sum_{i<=e} counts[i]
@triton.jit
def _inclusive_scan_kernel(counts_ptr, offsets_incl_ptr, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    acc = tl.zeros((), dtype=tl.int32)
    # single program iterates in chunks
    for i0 in range(0, NUM_EXPERTS, BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        mask = idx < NUM_EXPERTS
        cnt = tl.load(counts_ptr + idx, mask=mask, other=0)
        acc += tl.sum(cnt, axis=0)  # accumulate scalar acc
        tl.store(offsets_incl_ptr + idx, acc, mask=mask)


# Kernel 3: compute stable argsort permutation (sorted_token_indices) using base + tie-breaker.
# Each program handles one j. It computes rank = base_excl + tie_count, and stores j -> rank.
@triton.jit
def _stable_argsort_rank_kernel(
    flat_ptr, offsets_incl_ptr, out_perm_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr
):
    j = tl.program_id(axis=0)
    if j >= M:
        return
    # Load value at position j and its key
    val_j = tl.load(flat_ptr + j)
    key_j = val_j  # keys are in [0, NUM_EXPERTS)

    # Compute base_excl: inclusive prefix sum up to previous key (if any)
    if key_j == 0:
        base_excl = tl.zeros((), dtype=tl.int32)
    else:
        base_excl = tl.load(offsets_incl_ptr + key_j - 1)

    # Tie count: number of previous elements t < j with same key and smaller value
    tie_count = tl.zeros((), dtype=tl.int32)
    for t0 in range(0, j, BLOCK):
        idx = t0 + tl.arange(0, BLOCK)
        mask = (idx < j) & (idx >= 0)
        vals_t = tl.load(flat_ptr + idx, mask=mask, other=0)
        is_eq_key = vals_t == key_j
        is_less = vals_t < val_j
        # only consider idx < j
        valid = mask & is_eq_key & is_less
        tie_count += tl.sum(valid.to(tl.int32), axis=0)

    rank = base_excl + tie_count
    tl.store(out_perm_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts
        # Tunable block size; 1024 works well for typical M sizes
        self.block_size = 1024

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Computes sorted_token_indices: stable argsort permutation of indices by flat values.
        - Computes expert_offsets: length = num_experts + 1, inclusive cumulative counts per expert + 1.
        Returns (sorted_token_indices, expert_offsets).
        """
        # Ensure contiguity and device
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        device = flat.device
        NUM_EXPERTS = self.num_experts

        # 1) Compute per-expert counts using Triton
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        _count_experts_kernel[(NUM_EXPERTS,)](
            flat, counts, M, NUM_EXPERTS, self.block_size
        )

        # 2) Compute inclusive prefix sums of counts (offsets_incl[e] = sum_{i<=e} counts[i]) using Triton
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        _inclusive_scan_kernel[(1,)](
            counts, offsets_incl, NUM_EXPERTS, self.block_size
        )

        # 3) Compute sorted_token_indices (stable) using Triton
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        _stable_argsort_rank_kernel[(M,)](
            flat, offsets_incl, sorted_token_indices, M, NUM_EXPERTS, self.block_size
        )

        # 4) Compute expert_offsets as length = NUM_EXPERTS + 1, with +1 at the end
        #    Note: offsets_incl has length NUM_EXPERTS (inclusive prefix sums up to each expert).
        total = counts.sum()  # single torch reduction to get the last +1 correctly
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        # Fill first NUM_EXPERTS entries from offsets_incl
        expert_offsets[:NUM_EXPERTS] = offsets_incl  # inclusive prefix sums
        # Last entry: sum of all counts (+1)
        expert_offsets[NUM_EXPERTS] = total + 1

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
