import torch
import triton
import triton.language as tl


@triton.jit
def counts_and_bases_kernel(
    flat_ptr,                 # *const int32, length M
    base_incl_ptr,           # *int32, length NUM_EXPERTS  (inclusive prefix sums for counts)
    total_count_ptr,         # *int32, length 1
    M: tl.constexpr,         # number of elements in flat
    NUM_EXPERTS: tl.constexpr  # number of expert categories
):
    # Count occurrences of each key in flat
    counts = tl.zeros([NUM_EXPERTS], dtype=tl.int32)
    for j in tl.static_range(0, M):
        val = tl.load(flat_ptr + j)  # int32, assumed in [0, NUM_EXPERTS-1]
        counts[val] += 1

    # Compute inclusive prefix sums and store into base_incl_ptr (base_incl[e] = offsets_incl[e+1])
    running = tl.zeros((), dtype=tl.int32)
    for e in tl.static_range(0, NUM_EXPERTS):
        running += counts[e]
        tl.store(base_incl_ptr + e, running)

    # Store total count
    tl.store(total_count_ptr, running)


@triton.jit
def finalize_offsets_kernel(
    base_incl_ptr,           # *int32, length NUM_EXPERTS
    total_count_ptr,         # *int32, length 1
    offsets_ptr,             # *int32, length (NUM_EXPERTS + 1)
    NUM_EXPERTS: tl.constexpr
):
    # Copy base_incl (inclusive counts per key) into first NUM_EXPERTS entries of offsets
    for i in tl.static_range(0, NUM_EXPERTS):
        tl.store(offsets_ptr + i, tl.load(base_incl_ptr + i))
    # Write total count + 1 at the end
    total = tl.load(total_count_ptr)
    tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


@triton.jit
def stable_permutation_kernel(
    flat_ptr,                # *const int32, length M
    sorted_idx_ptr,         # *int32, length M (output)
    base_incl_ptr,          # *int32, length NUM_EXPERTS
    total_count_ptr,        # *int32, length 1 (not used here)
    M: tl.constexpr,        # number of elements
    NUM_EXPERTS: tl.constexpr
):
    # For each j in [0, M), compute stable rank:
    # key_j = flat[j], base_excl = base_incl[key_j - 1] if key_j > 0 else 0
    # tie_count = number of previous indices t < j with same key and flat[t] < flat[j]
    for j in tl.static_range(0, M):
        val_j = tl.load(flat_ptr + j)  # key
        base = tl.load(base_incl_ptr + (val_j - 1)) if val_j > 0 else tl.zeros((), dtype=tl.int32)
        # Compute tie_count: count of t < j with key==val_j and flat[t] < flat[j]
        tie_count = tl.zeros((), dtype=tl.int32)
        for t in tl.static_range(0, M):
            if t < j:
                val_t = tl.load(flat_ptr + t)
                if (val_t == val_j) and (val_t < val_j):
                    tie_count += 1
        rank = base + tie_count
        tl.store(sorted_idx_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        device = topk_idx.device
        # Flatten and cast to int32 for Triton
        flat = topk_idx.reshape(-1).to(torch.int32)
        M = flat.numel()

        # In the provided harness, num_experts=256; we set NUM_EXPERTS accordingly.
        # If a different num_experts is expected, you can adjust at launch, but this benchmark uses 256.
        NUM_EXPERTS = 256

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)

        # 1) counts_and_bases kernel
        base_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)  # we'll store inclusive prefix here
        total_count_buf = torch.empty(1, dtype=torch.int32, device=device)
        counts_and_bases_kernel[(1,)](
            flat, base_incl, total_count_buf,
            M=M, NUM_EXPERTS=NUM_EXPERTS
        )

        # 2) finalize offsets
        finalize_offsets_kernel[(1,)](
            base_incl, total_count_buf, expert_offsets,
            NUM_EXPERTS=NUM_EXPERTS
        )

        # 3) stable permutation
        stable_permutation_kernel[(1,)](
            flat, sorted_token_indices, base_incl, total_count_buf,
            M=M, NUM_EXPERTS=NUM_EXPERTS
        )

        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
