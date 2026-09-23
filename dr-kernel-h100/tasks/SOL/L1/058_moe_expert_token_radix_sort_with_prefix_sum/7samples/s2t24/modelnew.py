import torch
import triton


# Triton kernel: histogram of expert IDs using atomic_add
@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    # Load values; for masked lanes, provide 0 (won't contribute due to mask)
    vals = tl.load(x_ptr + offs, mask=mask, other=0)
    # Increment counts per value via atomic_add
    # Note: counts_ptr is int32, vals are int32
    for i in range(0, BLOCK):
        idx = offs[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + vals[i], 1)


# Triton kernel: compute inclusive prefix sum of counts into offsets[1..E]
@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Initialize offsets[0] on host before calling this kernel
    acc = 0
    # Loop over experts: offsets[i] = acc + counts[i]
    for i in range(0, E):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


# Triton kernel: stable counting sort producing sorted_token_indices permutation
@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Iterate positions in ascending order, BLOCK at a time
    for pos_start in range(0, N, BLOCK):
        pos = pos_start + tl.arange(0, BLOCK)
        mask = pos < N
        ids = tl.load(x_ptr + pos, mask=mask, other=0)  # int32
        # For each lane, write to out at starts[ids], then advance starts[ids]
        for i in range(0, BLOCK):
            p = pos[i]
            if mask[i]:
                idv = ids[i]
                idx = tl.load(starts_ptr + idv)
                tl.store(out_ptr + idx, p)
                # advance starts for this id
                tl.atomic_add(starts_ptr + idv, 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = int(num_experts)

    def forward(self, topk_idx: torch.Tensor):
        # Use provided topk_idx exactly; no torch operations in forward
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts  # 256 as per the original code

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0  # inclusive prefix sum will fill offsets[1..E]
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        BLOCK_SORT = 1024  # process positions in chunks
        grid_sort = (triton.cdiv(N, BLOCK_SORT),)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=BLOCK_SORT)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets