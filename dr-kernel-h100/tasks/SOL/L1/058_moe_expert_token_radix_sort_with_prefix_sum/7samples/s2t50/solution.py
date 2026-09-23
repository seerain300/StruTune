import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of x with mask
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0)  # assume int32 input

    # For each possible expert id v in [0, E), count occurrences in this block
    # Use masked atomic_add for safety
    for v in range(E):
        # Note: Triton will JIT this loop; ensure we don't access out-of-range indices
        v_vec = tl.full((BLOCK,), v, tl.int32)
        eq = (x_vals == v_vec) & mask  # boolean vector
        # Sum boolean vector to integer count; Triton supports tl.sum on boolean by casting
        cnt = tl.sum(eq.to(tl.int32))
        # Atomic add into counts[v]
        tl.atomic_add(counts_ptr + v, cnt)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute offsets[i+1] = offsets[i] + counts[i], offsets[0] = 0
    pid = tl.program_id(0)
    idx = pid  # single element per program for simplicity
    # Handle only idx < E; caller ensures grid >= E+1 and we use offsets[0]=0
    if idx < E:
        acc = tl.zeros((), dtype=tl.int32)
        # Loop over all j <= idx to compute inclusive sum
        for j in range(0, idx + 1):
            acc += tl.load(counts_ptr + j)
        tl.store(offsets_ptr + idx + 1, acc)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: iterate positions in order and write to out at starts[id]
    for pos in range(0, N):
        id_val = tl.load(x_ptr + pos)  # id_val is scalar int32
        # Load current start for this id and write position
        start = tl.load(starts_ptr + id_val)
        tl.store(out_ptr + start, pos)
        # Increment start for next token with same id (stable tie-break by position)
        tl.atomic_add(starts_ptr + id_val, 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Do not use any torch ops; use provided input tensor and Triton kernels
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        BLOCK_SCAN = 1
        grid_scan = (E,)  # one program per index up to E
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
