import torch

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        start = pid * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load flat values (int32), masked out-of-range as 0 (not used)
        vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

        # Ensure vals are in range [0, num_experts-1]; masked lanes have vals=0 and will not atomically contribute
        valid = (vals >= 0) & (vals < num_experts)
        vals = tl.where(valid, vals, 0)

        # Atomic add counts for valid lanes
        tl.atomic_add(counts_ptr + vals, 1, mask=valid)

    @triton.jit
    def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
        # Compute inclusive prefix sums for counts into offsets[0..num_experts]
        # We do it in a loop over experts, launching a single program instance (grid=1).
        # offsets_ptr length should be num_experts + 1.
        # offsets[0] = 0; for i in 1..num_experts: offsets[i] = offsets[i-1] + counts[i-1]
        # Note: This is simple and robust; given num_experts=256, it’s fast.
        # Triton doesn’t have built-in parallel scan, but we can do a sequential loop here.
        # We assume num_experts is passed and manageable.
        # Implementation: offsets_ptr is length num_experts+1; we start at 1
        # offsets_ptr[0] remains 0 since it’s unused; we write from index 1.

        # We use a loop over i in range(num_experts) and write to offsets[i+1]
        # However, Triton JIT requires static range; pass num_experts as tl.constexpr? Not applicable here.
        # So we use a while loop:
        i = 0
        running = tl.load(offsets_ptr + 0)  # read offsets[0] (should be 0)
        while i < num_experts:
            # Read counts[i]
            c = tl.load(counts_ptr + i)
            # offsets[i+1] = offsets[i] + counts[i]
            tl.store(offsets_ptr + (i + 1), running + c)
            running = running + c
            i += 1

    @triton.jit
    def _counting_sort_with_indices_kernel(flat_ptr, indices_ptr, offsets_ptr, out_indices_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
        # This kernel writes out sorted indices based on offsets. It assumes offsets_ptr length = num_experts + 1.
        # For each i in 0..n_elements-1: read val = flat[i]; pos = offsets[val]; write out_indices[pos] = indices[i].
        # We process the indices vector in chunks.
        pid = tl.program_id(axis=0)
        start = pid * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load original flat values and original indices (0..n_elements-1)
        vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
        idxs = tl.load(indices_ptr + offsets, mask=mask, other=0)  # indices 0..n_elements-1

        # Map each index to its sorted position using offsets
        # For masked-out lanes, we can compute but they won't store due to mask in stores.
        pos = tl.load(offsets_ptr + vals)  # offsets_ptr length should be >= num_experts+1, but we only need vals in [0, num_experts]

        # Store to out_indices at position pos
        tl.store(out_indices_ptr + pos, idxs, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # We must use Triton kernels; ensure we’re on CUDA
        if not TRITON_AVAILABLE or not topk_idx.is_cuda:
            # Fallback: use original PyTorch behavior (for safety)
            flat = topk_idx.reshape(-1)
            sorted_token_indices = torch.sort(flat, stable=True)[1].to(torch.int64)
            # Compute counts and offsets with torch for correctness
            counts = torch.bincount(flat.long(), minlength=self.num_experts)
            expert_offsets = torch.cumsum(counts, dim=0).to(torch.int32)
            return sorted_token_indices, expert_offsets

        # Triton-only path
        flat = topk_idx.reshape(-1)  # 1D int32
        N = flat.numel()
        # Make sure flat is int32 and contiguous
        if flat.dtype != torch.int32:
            flat_i32 = flat.to(torch.int32)
        else:
            flat_i32 = flat

        # Allocate counts and offsets
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat_i32.device)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat_i32.device)

        # Kernel 1: histogram
        BLOCK_SIZE = 1024
        grid_hist = (triton.cdiv(N, BLOCK_SIZE),)
        _histogram_counts_kernel[grid_hist](flat_i32, counts, N, self.num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # Kernel 2: inclusive prefix sum to get offsets
        # offsets[0] will be 0; we write offsets[1..num_experts] = cumsum(counts)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # Prepare indices and output for counting sort
        indices = torch.arange(N, dtype=torch.int32, device=flat_i32.device)
        out_indices = torch.empty(N, dtype=torch.int32, device=flat_i32.device)

        # Kernel 3: counting sort to produce sorted_token_indices permutation
        grid_sort = (triton.cdiv(N, BLOCK_SIZE),)
        _counting_sort_with_indices_kernel[grid_sort](flat_i32, indices, offsets, out_indices, N, self.num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # Return Triton-produced sorted indices (cast to int64 to match original) and expert_offsets
        sorted_token_indices = out_indices.to(torch.int64)
        return sorted_token_indices, offsets