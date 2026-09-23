import torch
import triton
import triton.language as tl


# Kernel 1: compute per-expert counts and their inclusive base positions (prefix sums).
@triton.jit
def _counts_and_bases_kernel(flat_ptr, counts_ptr, base_incl_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    # For each key k in [0, NUM_EXPERTS), count occurrences in flat_ptr
    for k in range(0, NUM_EXPERTS):
        cnt = tl.zeros((), dtype=tl.int32)
        for j in range(0, M):
            val = tl.load(flat_ptr + j)
            if val == k:
                cnt += 1
        tl.store(counts_ptr + k, cnt)

    # Inclusive prefix sum of counts -> base_incl_ptr
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        acc += tl.load(counts_ptr + i)
        tl.store(base_incl_ptr + i, acc)


# Kernel 2: produce stable sorted permutation using base positions and tie-breakers.
@triton.jit
def _stable_permutation_kernel(flat_ptr, base_incl_ptr, out_perm_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    # Compute for each j its rank (sorted position) stably.
    for j in range(0, M):
        val_j = tl.load(flat_ptr + j)   # value at j
        key_j = val_j                   # keys are in [0, NUM_EXPERTS)
        # base_excl = inclusive count up to previous key (if any)
        if key_j == 0:
            base_excl = tl.zeros((), dtype=tl.int32)
        else:
            base_excl = tl.load(base_incl_ptr + key_j - 1)
        # Tie count: how many elements before j have the same key and smaller value
        tie_count = tl.zeros((), dtype=tl.int32)
        for t in range(0, j):
            val_t = tl.load(flat_ptr + t)
            key_t = val_t
            if key_t == key_j and val_t < val_j:
                tie_count += 1
        rank = base_excl + tie_count
        tl.store(out_perm_ptr + j, rank)


# Kernel 3: compute expert offsets via inclusive scan (prefix sum) of counts
@triton.jit
def _inclusive_scan_kernel(counts_ptr, offsets_ptr, N: tl.int32):
    # Perform inclusive scan across N elements; write into offsets_ptr[1:], offset 0 = 0
    acc = tl.zeros((), dtype=tl.int32)
    # offset 0 is 0
    if N > 0:
        tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(0, N):
        acc += tl.load(counts_ptr + i)
        if (i + 1) < N:
            tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of run():
        - Computes sorted_token_indices: stable permutation of [0..M-1] ordered by topk_idx values (flat.sort(stable=True)).
        - Computes expert_offsets: length = num_experts + 1, inclusive cumulative counts per expert + 1.
        Returns: (sorted_token_indices, expert_offsets)
        """
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        NUM_EXPERTS = self.num_experts
        device = flat.device

        # Output buffers
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        base_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)

        # Launch Triton kernel 1: compute counts and inclusive base positions
        _counts_and_bases_kernel[(1,)](flat, counts, base_incl, M, NUM_EXPERTS)

        # Launch Triton kernel 2: compute stable permutation ranks
        _stable_permutation_kernel[(1,)](flat, base_incl, sorted_token_indices, M, NUM_EXPERTS)

        # Compute expert_offsets via Triton inclusive scan of counts
        _inclusive_scan_kernel[(1,)](counts, expert_offsets, NUM_EXPERTS)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
