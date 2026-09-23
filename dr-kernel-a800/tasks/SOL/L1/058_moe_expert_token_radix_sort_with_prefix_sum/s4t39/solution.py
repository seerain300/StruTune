import torch
import triton
import triton.language as tl


# Triton kernel: odd-even transposition sort for 1D int32 array.
# We sort the values in 'vals' and output the permutation 'indices_out' (i.e., the sorted_token_indices).
# This is a stable sort: equal keys keep original order.
# N is the length of the array; BLOCK is the tile size for iteration.
@triton.jit
def _odd_even_sort_stable(vals_ptr, indices_ptr, N, num_blocks: tl.constexpr, BLOCK: tl.constexpr):
    # We will perform N passes. In each pass:
    # even phase: compare-swap (0,1), (2,3), ...
    # odd phase:  compare-swap (1,2), (3,4), ...
    # Maintain 'indices' buffer: at the end, indices[i] = original position of sorted value at position i.
    for t in tl.static_range(0, N):
        is_even = (t % 2) == 0
        if is_even:
            # compare-swap pairs (0,1), (2,3), ...
            for i in tl.static_range(0, N, 2):
                a = i
                b = i + 1
                # Only process valid pairs
                if b < N:
                    ai = tl.load(vals_ptr + a)
                    bi = tl.load(vals_ptr + b)
                    ia = tl.load(indices_ptr + a)
                    ib = tl.load(indices_ptr + b)
                    # stable: if equal, keep original order (ai < bi or ai == bi)
                    swap = (ai > bi)
                    new_a = tl.where(swap, bi, ai)
                    new_b = tl.where(swap, ai, bi)
                    tl.store(vals_ptr + a, new_a)
                    tl.store(vals_ptr + b, new_b)
                    new_ia = tl.where(swap, ib, ia)
                    new_ib = tl.where(swap, ia, ib)
                    tl.store(indices_ptr + a, new_ia)
                    tl.store(indices_ptr + b, new_ib)
        else:
            # compare-swap pairs (1,2), (3,4), ...
            for i in tl.static_range(1, N, 2):
                a = i
                b = i + 1
                if b < N:
                    ai = tl.load(vals_ptr + a)
                    bi = tl.load(vals_ptr + b)
                    ia = tl.load(indices_ptr + a)
                    ib = tl.load(indices_ptr + b)
                    swap = (ai > bi)
                    new_a = tl.where(swap, bi, ai)
                    new_b = tl.where(swap, ai, bi)
                    tl.store(vals_ptr + a, new_a)
                    tl.store(vals_ptr + b, new_b)
                    new_ia = tl.where(swap, ib, ia)
                    new_ib = tl.where(swap, ia, ib)
                    tl.store(indices_ptr + a, new_ia)
                    tl.store(indices_ptr + b, new_ib)

# Triton kernel: histogram of flattened expert indices.
# Input:
#   topk_flat_ptr: pointer to int32 flattened indices (length N)
#   counts_ptr: pointer to int32 counts (length num_experts)
# We process the array in blocks and aggregate per-bin counts to minimize atomics.
@triton.jit
def _hist_kernel(topk_flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(topk_flat_ptr + offsets, mask=mask, other=0)
    # For each possible bin, accumulate how many times it appears in this block,
    # then emit a single atomic add per bin to the global counts.
    for b in tl.static_range(0, num_experts):
        # Mask: only count lanes where vals == b and within range
        eq = (vals == b) & mask
        # Sum booleans: cast to int32 and reduce
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        # Atomic add to global counts[b]
        tl.atomic_add(counts_ptr + b, cnt)


# Triton kernel: compute inclusive prefix sums of counts to produce expert_offsets.
# Input:
#   counts_ptr: pointer to int32 counts (length num_experts)
#   offsets_ptr: pointer to int32 offsets (length num_experts+1)
# We iterate j=0..num_experts-1 and accumulate.
@triton.jit
def _prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Single program computes prefix sums
    acc = tl.zeros((), dtype=tl.int32)
    # Write offset[0] = 0
    tl.store(offsets_ptr + 0, acc)
    for j in tl.static_range(0, num_experts):
        acc += tl.load(counts_ptr + j)
        tl.store(offsets_ptr + j + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure on CUDA and contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        topk_idx = topk_idx.contiguous()
        device = topk_idx.device

        # Flatten to 1D int32
        flat = topk_idx.reshape(-1).to(torch.int32)

        N = flat.numel()
        num_experts = 256  # fixed per provided configuration

        # 1) Triton odd-even transposition sort to produce sorted_token_indices (stable)
        # Allocate output indices buffer (same length as flat)
        indices_out = torch.empty_like(flat, dtype=torch.int32, device=device)

        # Launch Triton sort kernel
        # We choose BLOCK=1024 and num_blocks to tile over N. The kernel loops over N passes.
        BLOCK = 1024
        num_blocks = triton.cdiv(N, BLOCK)
        _odd_even_sort_stable[(num_blocks,)](
            flat, indices_out, N, num_blocks, BLOCK
        )

        # 2) Triton histogram to get per-expert counts
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_SIZE = 2048
        _hist_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
            flat, counts, N, num_experts, BLOCK_SIZE
        )

        # 3) Triton prefix-sum to produce expert_offsets (inclusive)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        _prefix_sum_kernel[(1,)](
            counts, offsets, num_experts
        )

        # Return sorted_token_indices and expert_offsets, matching original API
        # Note: sorted_token_indices are the positions in the original flattened order after stable sort.
        # However, the original run returns the permutation (indices) and offsets. We match that.
        # The stable sort ensures equal elements maintain original order.
        return indices_out, offsets


def run(*args):
    return ModelNew()(*args)
