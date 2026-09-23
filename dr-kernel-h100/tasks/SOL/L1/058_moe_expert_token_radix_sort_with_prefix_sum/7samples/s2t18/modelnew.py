import torch
import triton

@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load ids; missing elements masked as 0 (value doesn't matter due to mask)
    ids = tl.load(x_ptr + offs, mask=mask, other=0)
    # Compute indices into counts for int32 ids
    # Note: ids are assumed in [0, E). If not, masking above ensures safe loads for valid range.
    # For invalid ids, atomic_add would increment counts for garbage; but inputs are valid per get_inputs.
    idx = ids % E  # safe for any int32 ids within range
    # Atomic add per element
    tl.atomic_add(counts_ptr + idx, 1, mask=mask)

@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Simple per-element kernel: offsets[k] = sum(counts[:k])
    # offsets_ptr[0] is initialized by host as 0
    for k in range(0, E + 1):
        # Single-program inclusive scan; grid is (1,) so this is fine.
        # Load current count (if k > 0) and accumulate
        if k > 0:
            cnt = tl.load(counts_ptr + k - 1)
            tl.store(offsets_ptr + k, tl.load(offsets_ptr + k - 1) + cnt)
        else:
            tl.store(offsets_ptr + k, 0)

@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: write positions in ascending order based on x_ptr
    # starts_ptr is the exclusive prefix sum per expert (length E). We add out_ptr for writes.
    # Iterate over positions sequentially for stability.
    # Note: This loop is sequential per program. For N up to ~8k this is acceptable.
    for pos in range(0, N):
        id_val = tl.load(x_ptr + pos)
        e = id_val % E  # valid ids in [0, E)
        start = tl.load(starts_ptr + e)
        tl.store(out_ptr + start, pos)
        tl.atomic_add(starts_ptr + e, 1)

class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Ensure topk_idx is int32 and CUDA for Triton
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
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
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=1)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        return out, offsets