import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_rank_indices_kernel(flat_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # Each program handles a block of original indices i
    pid = tl.program_id(axis=0)
    # Vector of indices for this program
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    # Mask for valid i
    mask_i = idx < N

    # Initialize ranks for each i
    ranks = tl.zeros([BLOCK], dtype=tl.int32)

    # For each j, update ranks of all i
    # We iterate j from 0 to N-1 in steps; Triton supports dynamic loops.
    for j in range(0, N):
        # Skip if j is out of range (mask handled by N, but we keep it safe)
        # Load flat[j]; scalar load is fine
        val_j = tl.load(flat_ptr + j)

        # For each i in the block, update rank based on comparisons with val_j
        # Only consider i where mask_i is true
        # less: number of elements strictly less than val_j among all j
        # tie: number of elements equal to val_j with original index j < i
        # We compute these as counts and add to ranks
        # Use masked loads for flat[i] with mask_i; val for invalid i set to a large number to not affect count
        # However, we cannot load flat[i] for invalid i. Instead, we guard via mask_i.
        # The inner loop updates ranks only for valid i.

        # We need vectorized comparisons. Create a vector of val_j for broadcasting.
        # Triton supports elementwise comparison with scalar.
        # Compute less and tie for all i in the block
        # less: sum((flat[i] > val_j) & mask_i)
        # tie: sum((flat[i] == val_j) & (i > j) & mask_i)
        # Note: for invalid i (not mask_i), flat[i] is not loaded; but comparisons are safe because we only add for mask_i.

        # We need flat_i; since we don't load for invalid i, we instead compute using the mask:
        # We cannot directly load flat[i] for invalid i, so we avoid adding contributions for invalid i by masking.
        # To do so, we recompute less and tie by first computing the base condition and then masking by i_lt_j.

        # We'll compute i_lt_j vector: idx > j for tie; idx != j for comparison.
        i_lt_j = idx > j
        not_j = idx != j

        # Load flat[i] for valid i in this block
        flat_i = tl.load(flat_ptr + idx, mask=mask_i, other=0)

        # Compute less and tie counts for this j and this block
        less_i = (flat_i > val_j) & mask_i
        tie_i = (flat_i == val_j) & mask_i & i_lt_j

        # Reduce: count number of true per element in this block
        less_count = tl.sum(less_i.to(tl.int32), axis=0)
        tie_count = tl.sum(tie_i.to(tl.int32), axis=0)

        # Update ranks: increase for all valid i in the block
        ranks += less_count + tie_count

    # Now out[rank] = i. We only set for valid i; others remain untouched (zeros).
    tl.store(out_ptr + ranks, idx, mask=mask_i)


@triton.jit
def _histogram_kernel(in_ptr, N, hist_ptr, num_buckets: tl.constexpr):
    # One program per element, atomic add into corresponding bucket
    pid = tl.program_id(axis=0)
    val = tl.load(in_ptr + pid)
    # Ensure val is within [0, num_buckets-1]. In our use, get_inputs guarantees this.
    # Compute bucket index; cast to int32
    bucket = val.to(tl.int32)
    tl.atomic_add(hist_ptr + bucket, 1)


@triton.jit
def _inclusive_scan_prefix_sum(hist_ptr, out_ptr, num_buckets: tl.constexpr):
    # Single-program inclusive scan over a small fixed num_buckets (256)
    acc = 0
    for i in range(0, num_buckets):
        acc += tl.load(hist_ptr + i)
        tl.store(out_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and cast to int32
        flat = topk_idx.contiguous().view(-1).to(torch.int32)
        N = flat.numel()
        device = flat.device

        # 1) Triton stable argsort permutation: out[i] = index of i-th smallest value
        out = torch.empty(N, dtype=torch.int32, device=device)
        # Choose a moderate BLOCK to balance occupancy and simplicity
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _argsort_rank_indices_kernel[grid](flat, out, N, BLOCK=BLOCK)

        # 2) Triton histogram of expert IDs
        num_experts = 256  # matches original code
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_buckets=num_experts)

        # 3) Triton prefix sum to produce expert_offsets (inclusive), length num_experts + 1
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        # Return sorted permutation (int64) and expert_offsets (int32)
        return out.to(torch.int64), offsets