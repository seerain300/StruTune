import torch
import triton
import triton.language as tl


@triton.jit
def _counts_and_bases_kernel(flat_ptr, offsets_incl_ptr, total_count_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Compute per-expert counts via vectorized masked loads, then inclusive prefix sums (offsets_incl).
    Also compute total_count via final reduction and store in total_count_ptr.
    """
    # Chunk size for vectorized loads
    CHUNK = 256
    counts = tl.zeros(NUM_EXPERTS, dtype=tl.int32)

    # Loop over keys k and accumulate counts using chunked loads
    for k in range(0, NUM_EXPERTS):
        cnt = tl.zeros((), dtype=tl.int32)
        j = 0
        while j < M:
            idx = j + tl.arange(0, CHUNK)
            mask = idx < M
            vals = tl.load(flat_ptr + idx, mask=mask, other=0)
            # Count occurrences of k in this chunk: since Triton lacks direct count, loop over lanes
            for off in range(0, CHUNK):
                elem = vals[off]
                if mask[off]:
                    if elem == k:
                        cnt += 1
            j += CHUNK
        counts[k] = cnt

    # Inclusive prefix sums of counts -> offsets_incl
    acc = tl.zeros((), dtype=tl.int32)
    for e in range(0, NUM_EXPERTS):
        acc += counts[e]
        tl.store(offsets_incl_ptr + e, acc)

    # Store total count (single scalar)
    total = acc
    tl.store(total_count_ptr, total)


@triton.jit
def _finalize_offsets_kernel(offsets_incl_ptr, expert_offsets_ptr, total_count_ptr, NUM_EXPERTS: tl.constexpr):
    """
    Build expert_offsets:
    - expert_offsets[:NUM_EXPERTS] = offsets_incl[:]
    - expert_offsets[NUM_EXPERTS] = total_count_ptr[0] + 1
    """
    total = tl.load(total_count_ptr)
    for e in range(0, NUM_EXPERTS):
        val = tl.load(offsets_incl_ptr + e)
        tl.store(expert_offsets_ptr + e, val)
    tl.store(expert_offsets_ptr + NUM_EXPERTS, total + 1)


@triton.jit
def _stable_permutation_kernel(flat_ptr, offsets_incl_ptr, sorted_perm_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Compute stable sorted permutation indices:
    For each j in [0..M):
      key_j = flat[j]
      base_excl = offsets_incl_ptr[key_j - 1] if key_j > 0 else 0
      tie_count = number of t < j with key_t == key_j and flat[t] < flat[j]
      rank = base_excl + tie_count
      sorted_perm[j] = rank
    """
    for j in range(0, M):
        val_j = tl.load(flat_ptr + j)
        key_j = val_j  # keys are 0..NUM_EXPERTS-1
        if key_j == 0:
            base_excl = tl.zeros((), dtype=tl.int32)
        else:
            base_excl = tl.load(offsets_incl_ptr + key_j - 1)
        tie_count = tl.zeros((), dtype=tl.int32)
        for t in range(0, j):
            val_t = tl.load(flat_ptr + t)
            key_t = val_t
            if key_t == key_j and val_t < val_j:
                tie_count += 1
        rank = base_excl + tie_count
        tl.store(sorted_perm_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        NUM_EXPERTS = self.num_experts
        device = flat.device

        # Allocate outputs
        sorted_perm = torch.empty(M, dtype=torch.int32, device=device)
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        total_count = torch.empty(1, dtype=torch.int32, device=device)

        # Kernel 1: compute counts and inclusive prefix sums
        _counts_and_bases_kernel[(NUM_EXPERTS,)](flat, offsets_incl, total_count, M, NUM_EXPERTS)

        # Kernel 2: finalize expert_offsets
        _finalize_offsets_kernel[(NUM_EXPERTS,)](offsets_incl, expert_offsets, total_count, NUM_EXPERTS)

        # Kernel 3: stable permutation
        _stable_permutation_kernel[(M,)](flat, offsets_incl, sorted_perm, M, NUM_EXPERTS)

        # Return results matching original signatures: (sorted_token_indices, expert_offsets)
        return sorted_perm, expert_offsets


def run(*args):
    return ModelNew()(*args)
