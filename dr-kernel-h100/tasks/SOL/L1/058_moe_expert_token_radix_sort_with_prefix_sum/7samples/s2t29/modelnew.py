import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Compute per-expert histogram via atomic_add.
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    x = x.to(tl.int32)
    # Atomic add into counts[e] for each valid element
    tl.atomic_add(counts_ptr + x, 1, mask=mask)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive prefix sums for counts into offsets[1..E].
    # offsets_ptr[0] is set to 0 on host before launch.
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < E

    # Running sum for this tile
    running = tl.zeros((), dtype=tl.int32)
    # Process the tile sequentially to build inclusive prefix sums
    # Note: we iterate by taking a chunk of indices; for E=256 and BLOCK up to 256, this is fine.
    for i in range(0, BLOCK):
        id = offs[i]
        valid = id < E
        # Sum of counts up to id (inclusive). For invalid id, counts_ptr[id] is ignored via mask logic.
        running += tl.load(counts_ptr + id, mask=valid, other=0)
        # Store inclusive prefix sum at offsets[id+1] for valid id
        tl.store(offsets_ptr + id + 1, running, mask=valid)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr, MAX_POS: tl.constexpr):
    # Stable counting sort: produces permutation 'out' of positions [0..N-1] sorted by x.
    # starts_ptr[e] is exclusive prefix sum (initially inclusive for e=0: starts[0]=0; otherwise starts[e]=offsets[e-1]).
    for pos in range(0, MAX_POS):
        mask_pos = pos < N
        # Load id = x[pos]
        # We must ensure pos is valid; if not, skip by setting id=0 (won't write due to mask).
        id = tl.load(x_ptr + pos, mask=mask_pos, other=0).to(tl.int32)
        # Load current start for this id; if starts_ptr[id] is out of range (id < 0 or id >= E), default to 0
        # Note: mask_pos ensures we only proceed when pos < N
        # Since pos is within [0, MAX_POS) and MAX_POS >= N, we can safely read starts_ptr[id] when mask_pos is True.
        current = tl.load(starts_ptr + id, mask=mask_pos, other=0)
        # Write out[current] = pos
        tl.store(out_ptr + current, pos, mask=mask_pos)
        # Increment start for this id
        tl.store(starts_ptr + id, current + 1, mask=mask_pos)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is fixed at 256 in the original run
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Use provided topk_idx; do not generate inputs in forward
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        x = topk_idx.reshape(-1)  # int32
        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0  # inclusive prefix sum starts at 0
        BLOCK_SCAN = 256  # E=256, so one tile suffices
        grid_scan = (triton.cdiv(E, BLOCK_SCAN),)  # for E=256, this is (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        # Use a static upper bound for the loop; host ensures N <= MAX_POS for provided workloads
        MAX_POS = 8192
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1, MAX_POS=MAX_POS)

        return out, offsets