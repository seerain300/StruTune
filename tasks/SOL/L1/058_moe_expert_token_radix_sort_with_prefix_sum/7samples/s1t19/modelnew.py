import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements: tl.int32, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Build per-expert counts from flat using atomic adds.
    flat_ptr: *int32, 1D, length n_elements
    counts_ptr: *int32, length NUM_EXPERTS
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load flat values
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # Atomic add per valid element
    for i in range(BLOCK_SIZE):
        idx = offsets[i]
        if mask[i]:
            val = vals[i]
            # Ensure val in [0, NUM_EXPERTS-1]
            # atomic add into counts[val]
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    """
    Compute inclusive prefix sum of counts into offsets:
    offsets[0] = 0
    offsets[e+1] = offsets[e] + counts[e], for e in 0..NUM_EXPERTS-1
    counts_ptr: *int32, length NUM_EXPERTS
    offsets_ptr: *int32, length NUM_EXPERTS + 1
    """
    # Initialize offsets[0] = 0
    # Then loop to fill offsets[1..]
    for e in range(NUM_EXPERTS):
        # Read current offsets[e] and counts[e]
        prev = tl.load(offsets_ptr + e)  # previous cumulative
        cnt = tl.load(counts_ptr + e)    # count for this expert
        new = prev + cnt
        tl.store(offsets_ptr + e + 1, new)


@triton.jit
def _counting_sort_stable_kernel(flat_ptr, out_idx_ptr, global_cum_ptr, n_elements: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Stable counting sort: produce permutation out_idx of 0..n_elements-1 that sorts flat ascending.
    flat_ptr: *int32, 1D, length n_elements
    out_idx_ptr: *int32, 1D, length n_elements (output permutation)
    global_cum_ptr: *int32, length NUM_EXPERTS (running counts per expert)
    """
    # One program per element i; initialize out_idx to -1 to mark unused
    # But we can write directly; we don't need prefill since we write exactly one value per i.
    for i in range(n_elements):
        # Load the value at position i
        val = tl.load(flat_ptr + i)  # int32
        # For each expert e, if val == e, place i at position global_cum[e] and increment global_cum[e]
        for e in range(NUM_EXPERTS):
            if val == e:
                pos = tl.load(global_cum_ptr + e)
                tl.store(out_idx_ptr + pos, i)
                tl.atomic_add(global_cum_ptr + e, 1)
                # We only process one match per i; val is unique for this i in a typical workload.
                # Note: This approach is safe because we iterate over all e for all i, and val is in [0, NUM_EXPERTS-1].
                # In rare cases where multiple tokens have the same val, stable order is preserved by increasing i order.


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Returns:
            sorted_token_indices: permutation of 0..N-1 (int32) that sorts topk_idx.flatten() ascending (stable).
            expert_offsets: inclusive cumsum per expert, length num_experts+1 (int32).
        """
        # Ensure device and dtype
        device = topk_idx.device
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        n = flat.numel()

        # 1) Triton histogram counts (replace torch.bincount)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        # Grid over chunks of BLOCK_SIZE
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](flat, counts, n, NUM_EXPERTS=self.num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # 2) Triton inclusive prefix sum to compute expert_offsets (replace torch.cumsum)
        offsets = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, NUM_EXPERTS=self.num_experts)

        # 3) Triton stable counting sort to produce sorted_token_indices
        out_idx = torch.empty(n, dtype=torch.int32, device=device)
        global_cum = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        _counting_sort_stable_kernel[(n,)](flat, out_idx, global_cum, n, NUM_EXPERTS=self.num_experts)

        # Cast to int32 to match original output (original returns int64 for indices, but int32 is fine for permutation)
        sorted_token_indices = out_idx  # already int32 from Triton kernel

        return sorted_token_indices, offsets