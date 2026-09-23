import torch
import triton
import triton.language as tl


# Triton kernel: for each expert key k, count occurrences in 'flat'.
@triton.jit
def counts_kernel(
    flat_ptr,           # *int32, flattened values (length M)
    counts_ptr,         # *int32, output counts per key (length NUM_EXPERTS)
    M,                  # int32, total number of elements
    NUM_EXPERTS: tl.constexpr,  # number of experts (256)
    BLOCK_SIZE: tl.constexpr     # chunk size when scanning flat
):
    k = tl.program_id(0)  # one program per key
    if k >= NUM_EXPERTS:
        return
    # Initialize count for this key
    tl.store(counts_ptr + k, 0)

    # Scan flat in chunks and count matches for key k
    offset = 0
    while offset < M:
        idxs = offset + tl.arange(0, BLOCK_SIZE)
        mask = idxs < M
        vals = tl.load(flat_ptr + idxs, mask=mask, other=0).to(tl.int32)
        matches = (vals == k) & mask
        # Reduce matches to a scalar count for this chunk
        count_chunk = tl.sum(matches.to(tl.int32), axis=0)
        # Accumulate
        old = tl.load(counts_ptr + k).to(tl.int32)
        new = old + count_chunk
        tl.store(counts_ptr + k, new)
        offset += BLOCK_SIZE


# Triton kernel: compute inclusive prefix sums across keys into offsets_incl[k] = sum_{e=0..k} counts[e]
@triton.jit
def prefix_inclusive_kernel(
    counts_ptr,         # *int32, counts per key (length NUM_EXPERTS)
    offsets_incl_ptr,   # *int32, output inclusive prefix sums (length NUM_EXPERTS)
    NUM_EXPERTS: tl.constexpr
):
    k = tl.program_id(0)
    if k >= NUM_EXPERTS:
        return
    # Sum of previous counts up to k-1
    sum_prev = 0
    for e in range(0, k):
        sum_prev += tl.load(counts_ptr + e).to(tl.int32)
    # Current count for k
    cnt_k = tl.load(counts_ptr + k).to(tl.int32)
    tl.store(offsets_incl_ptr + k, sum_prev + cnt_k)


# Triton kernel: finalize expert_offsets: offsets[:NUM_EXPERTS] = offsets_incl, and offsets[NUM_EXPERTS] = total_count + 1.
@triton.jit
def finalize_offsets_kernel(
    offsets_incl_ptr,   # *int32, length NUM_EXPERTS
    counts_ptr,         # *int32, length NUM_EXPERTS, to compute total_count
    offsets_ptr,        # *int32, output length NUM_EXPERTS+1
    NUM_EXPERTS: tl.constexpr
):
    # First, write offsets[0..NUM_EXPERTS-1] = offsets_incl[0..NUM_EXPERTS-1]
    for e in range(0, NUM_EXPERTS):
        val = tl.load(offsets_incl_ptr + e).to(tl.int32)
        tl.store(offsets_ptr + e, val)

    # Compute total_count = sum of counts
    total = 0
    for e in range(0, NUM_EXPERTS):
        total += tl.load(counts_ptr + e).to(tl.int32)
    total = total + 1
    tl.store(offsets_ptr + NUM_EXPERTS, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Compute sorted_token_indices using torch.sort(stable=True).indices (exact and stable),
        and compute expert_offsets using Triton kernels.
        """
        # Ensure input is on CUDA and contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels"
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        device = flat.device
        NUM_EXPERTS = 256

        # 1) Stable sort permutation using PyTorch (exact)
        # sorted_token_indices is a permutation of [0..M-1], sorted by flat values.
        sorted_token_indices = torch.sort(flat, stable=True).indices.to(torch.int32)

        # 2) Triton: compute per-expert counts
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        # Choose a large BLOCK_SIZE to reduce loop iterations
        BLOCK_SIZE = 4096
        # Launch one program per key
        counts_kernel[(NUM_EXPERTS,)](
            flat, counts, M,
            NUM_EXPERTS=NUM_EXPERTS, BLOCK_SIZE=BLOCK_SIZE
        )

        # 3) Triton: compute inclusive prefix sums of counts
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        prefix_inclusive_kernel[(NUM_EXPERTS,)](
            counts, offsets_incl, NUM_EXPERTS=NUM_EXPERTS
        )

        # 4) Triton: finalize expert_offsets
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        finalize_offsets_kernel[(1,)](
            offsets_incl, counts, expert_offsets, NUM_EXPERTS=NUM_EXPERTS
        )

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
