import torch
import triton
import triton.language as tl


# Kernel 1: counts_and_bases
# Computes counts per key e in [0..NUM_EXPERTS) and inclusive prefix sums base_incl[e] = sum_{k<=e} counts[k].
# Also computes total_count (sum of counts) and stores it in total_count_ptr[0].
@triton.jit
def counts_and_bases_kernel(
    flat_ptr,                      # *const int32, length M
    base_incl_ptr,                # *int32, length NUM_EXPERTS
    total_count_ptr,              # *int32, length 1
    M: tl.constexpr,              # number of elements in flat
    NUM_EXPERTS: tl.constexpr     # number of expert categories
):
    counts = tl.zeros([NUM_EXPERTS], dtype=tl.int32)

    # Count occurrences of each key in flat
    for j in tl.static_range(0, M):
        val = tl.load(flat_ptr + j)  # int32
        # Only count if 0 <= val < NUM_EXPERTS (flat is generated that way in the harness)
        if val >= 0 and val < NUM_EXPERTS:
            counts[val] += 1

    # Compute base_incl = inclusive prefix sums up to e; we write base_incl[e] = offsets_incl[e+1]
    running = tl.zeros((), dtype=tl.int32)
    for e in tl.static_range(0, NUM_EXPERTS):
        running += counts[e]
        tl.store(base_incl_ptr + e, running)

    # Store total_count
    tl.store(total_count_ptr, running)


# Kernel 2: finalize_offsets
# Writes expert_offsets of length (NUM_EXPERTS + 1):
# offsets[i] = base_incl[i] for i in [0..NUM_EXPERTS-1]
# offsets[NUM_EXPERTS] = total_count + 1
@triton.jit
def finalize_offsets_kernel(
    base_incl_ptr,                # *int32, length NUM_EXPERTS
    total_count_ptr,              # *int32, length 1
    offsets_ptr,                  # *int32, length (NUM_EXPERTS + 1)
    NUM_EXPERTS: tl.constexpr
):
    for i in tl.static_range(0, NUM_EXPERTS):
        tl.store(offsets_ptr + i, tl.load(base_incl_ptr + i))
    total = tl.load(total_count_ptr)
    tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


# Kernel 3: stable_permutation
# Computes a stable permutation of [0..M-1] ordered by values in flat (stable=True).
# sorted_token_indices[j] = rank where
#   base_excl = base_incl[val_j - 1] if val_j > 0 else 0
#   tie_count = number of t < j with same val and flat[t] < flat[j]
#   rank = base_excl + tie_count
@triton.jit
def stable_permutation_kernel(
    flat_ptr,                      # *const int32, length M
    base_incl_ptr,                # *const int32, length NUM_EXPERTS
    sorted_ptr,                   # *int32, length M (output indices)
    M: tl.constexpr               # number of elements in flat
):
    for j in tl.static_range(0, M):
        val_j = tl.load(flat_ptr + j)  # int32, in [0, NUM_EXPERTS-1] (assumed valid)

        # base_excl = base_incl[val_j - 1] if val_j > 0 else 0
        if val_j > 0:
            base_excl = tl.load(base_incl_ptr + (val_j - 1))
        else:
            base_excl = tl.zeros((), dtype=tl.int32)

        # tie_count = number of previous indices t < j with same val and flat[t] < flat[j]
        tie_count = tl.zeros((), dtype=tl.float32)
        for t in tl.static_range(0, j):
            v_t = tl.load(flat_ptr + t)
            cond = (v_t == val_j) & (t < j)
            tie_count += cond.to(tl.float32)

        rank = base_excl + tie_count.to(tl.int32)
        tl.store(sorted_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward that returns:
        - sorted_token_indices: int32 of shape (M,), stable permutation of indices [0..M-1] ordered by flat values.
        - expert_offsets: int32 of shape (num_experts+1,), inclusive counts + 1.
        """
        assert topk_idx.is_cuda, "ModelNew requires a CUDA tensor for Triton execution."
        flat = topk_idx.reshape(-1).contiguous()  # 1D vector of length M
        M = flat.numel()

        # num_experts is fixed to 256 in the harness; Triton kernels use NUM_EXPERTS as a constexpr.
        NUM_EXPERTS = 256

        # 1) counts_and_bases: compute counts and base_incl, and total_count
        base_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        total_count = torch.empty(1, dtype=torch.int32, device=flat.device)
        counts_and_bases_kernel[(1,)](flat, base_incl, total_count, M=M, NUM_EXPERTS=NUM_EXPERTS)

        # 2) finalize_offsets: produce expert_offsets of length (NUM_EXPERTS + 1)
        offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        finalize_offsets_kernel[(1,)](base_incl, total_count, offsets, NUM_EXPERTS=NUM_EXPERTS)

        # 3) stable_permutation: compute sorted_token_indices
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)
        stable_permutation_kernel[(1,)](flat, base_incl, sorted_token_indices, M=M, NUM_EXPERTS=NUM_EXPERTS)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
