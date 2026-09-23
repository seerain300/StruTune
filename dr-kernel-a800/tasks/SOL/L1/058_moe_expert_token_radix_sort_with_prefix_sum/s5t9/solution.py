import torch
import triton
import triton.language as tl


@triton.jit
def _counts_and_bases_kernel(flat_ptr, counts_ptr, offsets_incl_ptr, total_count_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    For each key k in [0, NUM_EXPERTS):
      counts[k] = number of elements in flat_ptr equal to k
    Then compute inclusive prefix sums:
      offsets_incl[e] = sum_{i=0..e} counts[i]
    Also write total_count = sum_{i=0..NUM_EXPERTS-1} counts[i] to total_count_ptr.
    """
    # Accumulate counts per key
    for k in range(0, NUM_EXPERTS):
        cnt = tl.zeros((), dtype=tl.int32)
        for j in range(0, M):
            val = tl.load(flat_ptr + j)
            if val == k:
                cnt += 1
        tl.store(counts_ptr + k, cnt)

    # Inclusive prefix sum of counts -> offsets_incl
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_incl_ptr + i, acc)

    # Compute total_count
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        total += tl.load(counts_ptr + i)
    tl.store(total_count_ptr, total)


@triton.jit
def _stable_permutation_kernel(flat_ptr, offsets_incl_ptr, out_perm_ptr, total_count_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Stable argsort by values in flat_ptr. The values are in [0..NUM_EXPERTS-1].
    For each j in [0, M):
      key_j = flat[j]
      base_excl = offsets_incl[key_j - 1] if key_j > 0 else 0
      tie_count = number of previous t < j with same key and flat[t] < flat[j]
      rank = base_excl + tie_count
      out_perm[j] = rank
    """
    for j in range(0, M):
        val_j = tl.load(flat_ptr + j)  # value (key) at position j
        key_j = val_j
        if key_j > 0:
            base_excl = tl.load(offsets_incl_ptr + key_j - 1)
        else:
            base_excl = tl.zeros((), dtype=tl.int32)
        tie_count = tl.zeros((), dtype=tl.int32)
        # Loop over all previous indices t < j to compute stable tie count
        for t in range(0, j):
            val_t = tl.load(flat_ptr + t)
            if (key_j == val_t) and (val_t < val_j):
                tie_count += 1
        rank = base_excl + tie_count
        tl.store(out_perm_ptr + j, rank)


@triton.jit
def _finalize_offsets_kernel(offsets_incl_ptr, total_count_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    """
    Copy inclusive prefix sums to offsets[:NUM_EXPERTS], and set offsets[NUM_EXPERTS] = total_count + 1.
    """
    for i in range(0, NUM_EXPERTS):
        tl.store(offsets_ptr + i, tl.load(offsets_incl_ptr + i))
    total = tl.load(total_count_ptr)
    tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - sorted_token_indices: stable argsort by flattened values of topk_idx (i.e., topk_idx.reshape(-1))
        - expert_offsets: inclusive prefix counts per expert plus 1
        Returns: (sorted_token_indices, expert_offsets)
        """
        # Flatten and ensure int32 values in expected range
        flat = topk_idx.reshape(-1)
        # Make sure dtype is int32 (PyTorch randint produced int32)
        flat = flat.to(torch.int32)

        M = flat.numel()
        NUM_EXPERTS = self.num_experts
        device = flat.device

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        # Allocate temporary buffers
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        total_count = torch.empty(1, dtype=torch.int32, device=device)  # scalar buffer for total_count

        # Kernel 1: compute counts and inclusive prefix sums, and total_count
        _counts_and_bases_kernel[(1,)](flat, counts, offsets_incl, total_count, M, NUM_EXPERTS)

        # Kernel 2: compute stable permutation ranks
        _stable_permutation_kernel[(1,)](flat, offsets_incl, sorted_token_indices, total_count, M, NUM_EXPERTS)

        # Kernel 3: finalize expert_offsets
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        _finalize_offsets_kernel[(1,)](offsets_incl, total_count, expert_offsets, NUM_EXPERTS)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
