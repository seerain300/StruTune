import torch
import triton

@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles a block of elements and atomically adds to counts
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    mask = idx < N
    # Load values; out-of-bounds elements masked to 0 (won't affect counts)
    ids = tl.load(x_ptr + idx, mask=mask, other=0).to(tl.int32)
    # Atomically accumulate counts for each id
    # Note: tl.atomic_add on vector assumes unique indices; here we use mask to skip OOB
    for i in range(BLOCK):
        val = ids[i]
        if val >= 0 and val < E and mask[i]:
            tl.atomic_add(counts_ptr + val, 1)

@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive prefix sum: offsets[i] = sum(counts[:i])
    # Initialize offsets[0] = 0 on host before launching.
    # Each program handles a block of elements and updates its local sum.
    pid = tl.program_id(0)
    start = pid * BLOCK
    # First pass: compute local sums for this block
    local_sum = tl.zeros((), dtype=tl.int32)
    for j in range(start, tl.minimum(start + BLOCK, E)):
        c = tl.load(counts_ptr + j)
        local_sum += c
        # offsets[j] = sum of previous elements
        tl.store(offsets_ptr + j, local_sum)
    # No barrier needed here; grid covers all elements

@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable sort: iterate positions in ascending order, place into out at starts[id], then advance starts[id]
    pos = 0  # scalar loop; ensures stability and bounds safety
    while pos < N:
        # read id at position pos
        id_val = tl.load(x_ptr + pos).to(tl.int32)
        if id_val >= 0 and id_val < E:
            start = tl.load(starts_ptr + id_val).to(tl.int32)
            tl.store(out_ptr + start, pos)
            tl.store(starts_ptr + id_val, start + 1)
        pos += 1

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is fixed to 256, matching the original run
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Use provided input exactly; no torch ops in forward
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 2048
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        # Simple per-element inclusive scan; grid covers all elements
        grid_scan = (E,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums: starts[e] = offsets[e]
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets