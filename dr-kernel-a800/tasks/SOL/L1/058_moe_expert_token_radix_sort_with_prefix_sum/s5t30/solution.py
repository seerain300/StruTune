import torch
import triton
import triton.language as tl


@triton.jit
def counts_and_bases_kernel(
    flat_ptr,                # *int32, flattened input
    counts_ptr,              # *int32, length NUM_EXPERTS
    offsets_incl_ptr,        # *int32, length NUM_EXPERTS
    total_count_ptr,         # *int32, scalar to hold total count
    M: tl.constexpr,         # int, total number of elements in flat
    NUM_EXPERTS: tl.constexpr,  # int, number of possible keys
    BLOCK: tl.constexpr,     # int, loop chunk size
):
    # Each program handles one key k in [0, NUM_EXPERTS)
    k = tl.program_id(0)

    # Accumulate count for this key across flat
    count = tl.zeros((), dtype=tl.int32)
    for start in range(0, M, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < M
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Count how many equal to k in this chunk
        count += tl.sum((vals == k) & mask, axis=0)
    tl.store(counts_ptr + k, count)

    # Compute inclusive prefix sums for offsets_incl[e] = sum_{t=0..e} counts[t]
    # Initialize offset with current count
    offset = count
    tl.store(offsets_incl_ptr + k, offset)

    # If k > 0, accumulate previous offsets_incl to get inclusive sum
    if k > 0:
        prev_sum = tl.zeros((), dtype=tl.int32)
        for t in range(0, k):
            prev_sum += tl.load(offsets_incl_ptr + t)
        offset += prev_sum
        tl.store(offsets_incl_ptr + k, offset)

    # Accumulate total count for device scalar
    total_count = tl.zeros((), dtype=tl.int32)
    for t in range(0, NUM_EXPERTS):
        total_count += tl.load(counts_ptr + t)
    tl.store(total_count_ptr, total_count)


@triton.jit
def stable_permutation_kernel(
    flat_ptr,                    # *int32, flattened input
    sorted_ptr,                  # *int32, output permutation indices
    offsets_incl_ptr,            # *int32, length NUM_EXPERTS, inclusive scan of counts
    M: tl.constexpr,             # int, total number of elements in flat
    NUM_EXPERTS: tl.constexpr,   # int, number of possible keys
):
    # One program per position j
    j = tl.program_id(0)
    # Load value for position j
    val_j = tl.load(flat_ptr + j)

    # Compute base exclusive prefix for key val_j: sum_{t < val_j} counts[t]
    base_excl = tl.zeros((), dtype=tl.int32)
    if val_j > 0:
        # Sum offsets_incl[0..val_j-1]
        for t in range(0, val_j):
            base_excl += tl.load(offsets_incl_ptr + t)

    # Compute tie_count: number of previous elements t < j with same key and flat[t] < flat[j]
    tie_count = tl.zeros((), dtype=tl.int32)
    for t in range(0, j):
        v = tl.load(flat_ptr + t)
        if v == val_j:
            tie_count += 1  # original index order is stable

    rank = base_excl + tie_count
    tl.store(sorted_ptr + j, rank)


@triton.jit
def finalize_offsets_kernel(
    offsets_incl_ptr,        # *int32, length NUM_EXPERTS
    total_count_ptr,         # *int32, scalar total count
    expert_offsets_ptr,      # *int32, output length (NUM_EXPERTS + 1)
    NUM_EXPERTS: tl.constexpr,
):
    # Write inclusive scan into first NUM_EXPERTS positions
    for e in range(0, NUM_EXPERTS):
        tl.store(expert_offsets_ptr + e, tl.load(offsets_incl_ptr + e))
    # Write total_count + 1 at the end
    total = tl.load(total_count_ptr)
    tl.store(expert_offsets_ptr + NUM_EXPERTS, total + 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton implementation of:
        - sorted_token_indices: permutation of [0..M-1] with stable ascending order by values in flat (ties by original index).
        - expert_offsets: inclusive count per expert plus 1 at the end.
        """
        assert topk_idx.is_cuda, "Input tensor must be on CUDA for Triton execution."
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()

        M = flat.numel()
        NUM_EXPERTS = 256  # fixed in the provided setup

        # 1) Compute counts per key and base inclusive prefix sums (and total_count)
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        total_count = torch.empty(1, dtype=torch.int32, device=flat.device)

        BLOCK = 1024  # chunk size for counts kernel scan
        grid_counts = (NUM_EXPERTS,)
        counts_and_bases_kernel[grid_counts](
            flat,
            counts,
            offsets_incl,
            total_count,
            M,
            NUM_EXPERTS,
            BLOCK,
        )

        # 2) Compute stable permutation indices (sorted_token_indices)
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)
        grid_perm = (M,)
        stable_permutation_kernel[grid_perm](
            flat,
            sorted_token_indices,
            offsets_incl,
            M,
            NUM_EXPERTS,
        )

        # 3) Finalize expert_offsets = inclusive scan + 1 at end
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        finalize_offsets_kernel[(1,)](
            offsets_incl,
            total_count,
            expert_offsets,
            NUM_EXPERTS,
        )

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
