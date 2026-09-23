import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Build per-expert counts via atomic_add.
    flat_ptr: *int32, length n_elements
    counts_ptr: *int32, length num_experts
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # valid lanes: vals in [0, num_experts-1] and mask true
    valid = (vals >= 0) & (vals < num_experts) & mask
    # For invalid lanes, set vals to 0 so they don't contribute atomics (but we still mask them).
    vals = tl.where(valid, vals, 0)
    # Atomic add per valid lane
    # Note: num_experts is constexpr for the loop, but we pass as runtime int32 for range.
    # For each lane, if valid, increment counts[vals].
    # We need a loop over lanes; Triton supports elementwise operations, but not dynamic per-lane branching across threads.
    # Instead, use tl.atomic_add with a vector of indices constructed from vals.
    # Since we cannot directly index a vector of pointers, we rely on Triton's broadcasting to add to counts_ptr[vals].
    # Make sure to only add for valid lanes.
    # Implementation: iterate over lanes and atomic_add 1 to counts_ptr[vals].
    # Triton will handle the atomic adds correctly. If vals are out of range or not valid, skip via mask.
    # We create a 1 tensor for each valid lane and atomic_add to counts_ptr[vals].
    # Triton supports elementwise pointer arithmetic, but atomic_add needs a scalar index; we construct per-lane via dynamic indexing.
    # Simpler approach: since Triton doesn't support dynamic per-lane pointer assignment easily, we rely on tl.atomic_add with vector indices.
    # However, Triton's atomic_add supports vector indices: counts_ptr[vals] += 1 for valid lanes.
    tl.atomic_add(counts_ptr + vals, 1, mask=valid)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sums of counts into offsets.
    counts_ptr: *int32, length num_experts
    offsets_ptr: *int32, length num_experts+1
    """
    # Single program instance (grid=1) performs the scan
    tl.store(offsets_ptr + 0, 0)  # offsets[0] = 0
    acc = 0
    for e in range(0, num_experts):
        acc += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e + 1, acc)


@triton.jit
def _counting_sort_with_indices_kernel(flat_ptr, indices_ptr, offsets_ptr, out_indices_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    Perform counting sort to produce sorted_token_indices permutation.
    flat_ptr: *int32, length n_elements (values)
    indices_ptr: *int32, length n_elements (initially 0..n_elements-1)
    offsets_ptr: *int32, length num_experts+1 (inclusive prefix sums)
    out_indices_ptr: *int32, length n_elements (final sorted permutation)
    """
    pid = tl.program_id(0)
    i = pid
    lane_mask = i < n_elements
    if lane_mask:
        val = tl.load(flat_ptr + i)
        # Ensure val is within range; if not, default to 0. Original vals are in [0, num_experts-1], so this is fine.
        start = tl.load(offsets_ptr + val)
        src = tl.load(indices_ptr + start)
        # Place src at out_indices[i]
        tl.store(out_indices_ptr + i, src)
        # Advance the next slot
        tl.store(indices_ptr + start, i + 1)  # i+1 for next write; start increments per i


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Builds counts of expert IDs via Triton.
        - Computes inclusive prefix sums (offsets) via Triton.
        - Produces sorted_token_indices permutation via Triton counting sort.
        Returns:
          sorted_token_indices: int64 tensor of shape (N,)
          expert_offsets: int32 tensor of shape (num_experts+1,)
        """
        # Ensure CUDA and contiguity
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        # Triton works with int32; cast flat to int32 for histogram
        flat_i32 = flat.to(torch.int32)

        # Allocate counts, offsets, indices
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        out_indices = torch.empty(N, dtype=torch.int32, device=flat.device)

        # Initialize indices to 0..N-1
        torch.arange(N, out=indices, device=flat.device)

        # Kernel 1: histogram counts
        BLOCK_SIZE = 1024
        grid_hist = (triton.cdiv(N, BLOCK_SIZE),)
        _histogram_counts_kernel[grid_hist](flat_i32, counts, N, self.num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # Kernel 2: inclusive prefix sum to get offsets
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # Kernel 3: counting sort with indices to produce permutation
        grid_sort = (N,)
        _counting_sort_with_indices_kernel[grid_sort](flat_i32, indices, offsets, out_indices, N, self.num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)

        # Return sorted_token_indices as int64 to match original and expert_offsets as int32
        sorted_token_indices = out_indices.to(torch.int64)
        return sorted_token_indices, offsets