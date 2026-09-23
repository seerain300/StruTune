import torch
import triton
import triton.language as tl


# Kernel A: compute per-expert counts and inclusive prefix sums (base positions), and total count.
@triton.jit
def _counts_and_bases_kernel(flat_ptr, counts_ptr, base_incl_ptr, total_count_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    For each key k in [0, NUM_EXPERTS):
      counts_ptr[k] = number of elements in flat_ptr equal to k
    Then compute base_incl_ptr[e] = inclusive prefix sum of counts[0..e].
    Also store total_count = sum(counts) into total_count_ptr[0].
    """
    # Accumulate counts per key
    for k in range(0, NUM_EXPERTS):
        cnt = tl.zeros((), dtype=tl.int32)
        for j in range(0, M):
            val = tl.load(flat_ptr + j)  # load int32 value
            if val == k:
                cnt += 1
        tl.store(counts_ptr + k, cnt)

    # Inclusive prefix sum of counts
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        acc += tl.load(counts_ptr + i)
        tl.store(base_incl_ptr + i, acc)

    # Total count is the last accumulated acc
    tl.store(total_count_ptr, acc)


# Kernel B: produce stable sorted permutation indices using base positions and tie-breakers.
@triton.jit
def _stable_permutation_kernel(flat_ptr, base_incl_ptr, out_perm_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    For each j in [0, M):
      key_j = flat[j]
      base_excl = offsets_incl[key_j - 1] if key_j > 0 else 0
      tie_count = number of t < j with same key and flat[t] < flat[j]
      rank = base_excl + tie_count
      out_perm[j] = rank
    """
    for j in range(0, M):
        val_j = tl.load(flat_ptr + j)
        key_j = val_j
        # base exclusive for this key: inclusive sum up to previous key
        if key_j == 0:
            base_excl = tl.zeros((), dtype=tl.int32)
        else:
            base_excl = tl.load(base_incl_ptr + key_j - 1)
        # tie count for stability: count elements before j with same key and smaller value
        tie_count = tl.zeros((), dtype=tl.int32)
        for t in range(0, j):
            val_t = tl.load(flat_ptr + t)
            key_t = val_t
            if key_t == key_j and val_t < val_j:
                tie_count += 1
        rank = base_excl + tie_count
        tl.store(out_perm_ptr + j, rank)


# Kernel C: finalize expert_offsets using the counts (inclusive prefix sums) and total count.
@triton.jit
def _finalize_offsets_kernel(counts_ptr, base_incl_ptr, offsets_ptr, total_count_ptr, NUM_EXPERTS: tl.constexpr):
    """
    offsets[:NUM_EXPERTS] = base_incl[:NUM_EXPERTS] (inclusive prefix sums of counts)
    offsets[NUM_EXPERTS] = tl.load(total_count_ptr) + 1
    """
    # Copy inclusive prefix sums
    for i in range(0, NUM_EXPERTS):
        tl.store(offsets_ptr + i, tl.load(base_incl_ptr + i))
    # Set the last element to total_count + 1
    total = tl.load(total_count_ptr) + 1
    tl.store(offsets_ptr + NUM_EXPERTS, total)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Ensure flat is int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)
        M = flat.numel()
        NUM_EXPERTS = self.num_experts
        device = flat.device

        # 1) Compute counts, inclusive base positions, and total_count using Triton
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        base_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        total_count = torch.empty(1, dtype=torch.int32, device=device)

        _counts_and_bases_kernel[(1,)](flat, counts, base_incl, total_count, M, NUM_EXPERTS)

        # 2) Compute stable sorted_token_indices via Triton
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        _stable_permutation_kernel[(1,)](flat, base_incl, sorted_token_indices, M, NUM_EXPERTS)

        # 3) Finalize expert_offsets via Triton (inclusive scan prefix and +1 at end)
        offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        _finalize_offsets_kernel[(1,)](counts, base_incl, offsets, total_count, NUM_EXPERTS)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
