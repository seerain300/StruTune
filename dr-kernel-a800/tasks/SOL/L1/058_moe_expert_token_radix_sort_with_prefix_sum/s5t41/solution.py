import torch
import triton
import triton.language as tl


# Kernel: count occurrences of each expert key in flat via atomic_add
@triton.jit
def count_per_exp_kernel(flat_ptr, counts_ptr, M):
    # Each program handles one element j, atomically adds 1 to counts[flat[j]]
    j = tl.program_id(0)
    if j < M:
        val_j = tl.load(flat_ptr + j)  # int32
        # Atomic add 1 to counts[val_j]
        tl.atomic_add(counts_ptr + val_j, 1)


# Kernel: inclusive prefix sums of counts -> offsets_incl
@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_incl_ptr, NUM_EXPERTS: tl.constexpr):
    e = tl.program_id(0)  # [0..NUM_EXPERTS-1]
    if e == 0:
        tl.store(offsets_incl_ptr + 0, tl.load(counts_ptr + 0))
    else:
        tl.store(offsets_incl_ptr + e, tl.load(offsets_incl_ptr + e - 1) + tl.load(counts_ptr + e))


# Kernel: stable argsort by values in flat, producing sorted_token_indices (permutation of [0..M-1])
@triton.jit
def stable_argsort_kernel(flat_ptr, sorted_idx_ptr, offsets_incl_ptr, M, NUM_EXPERTS: tl.constexpr):
    j = tl.program_id(0)  # thread id over j in [0..M)
    if j >= M:
        return
    val_j = tl.load(flat_ptr + j)  # int32
    # base_excl: inclusive prefix sum up to (val_j - 1), or 0 if val_j == 0
    base_excl = tl.zeros((), dtype=tl.int32)
    if val_j > 0:
        base_excl = tl.load(offsets_incl_ptr + (val_j - 1))
    else:
        base_excl = 0
    # tie_count: number of previous indices t < j with same key
    tie_count = tl.zeros((), dtype=tl.int32)
    # Scan all previous indices; count equal keys that come before j (preserves stable order)
    for t in range(0, M):
        if t < j:
            v_t = tl.load(flat_ptr + t)
            tie_count += tl.where(v_t == val_j, 1, 0)
    rank = base_excl + tie_count
    tl.store(sorted_idx_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure on CUDA device and contiguous
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        NUM_EXPERTS = 256
        assert flat.dtype == torch.int32, "flat must be int32."

        # 1) Compute counts per expert using Triton (atomic per element)
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        # Launch one program per element j
        grid_counts = (M,)
        count_per_exp_kernel[grid_counts](flat, counts, M)

        # 2) Inclusive prefix sums for offsets_incl
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        grid_scan = (NUM_EXPERTS,)
        prefix_sum_kernel[grid_scan](counts, offsets_incl, NUM_EXPERTS)

        # 3) Finalize expert_offsets: offsets[:NUM_EXPERTS] = offsets_incl, offsets[NUM_EXPERTS] = sum(counts) + 1
        # Compute sum(counts) in Python and construct offsets
        total_count = int(counts.sum().item()) + 1  # +1 per original code
        offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        # Fill offsets[:NUM_EXPERTS] = offsets_incl
        for i in range(NUM_EXPERTS):
            offsets[i] = offsets_incl[i]
        # Final offset at NUM_EXPERTS
        offsets[NUM_EXPERTS] = total_count

        # 4) Stable argsort using Triton: produce sorted_token_indices (permutation of [0..M-1])
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)
        stable_argsort_kernel[(M,)](flat, sorted_token_indices, offsets_incl, M, NUM_EXPERTS)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
