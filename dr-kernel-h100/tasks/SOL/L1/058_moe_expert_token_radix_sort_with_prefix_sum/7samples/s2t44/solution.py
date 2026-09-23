import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles a block of elements and atomically adds to counts
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    mask = idx < N
    # Load values safely; masked positions won't affect counts
    vals = tl.load(x_ptr + idx, mask=mask, other=0)
    # Atomically add 1 for each valid element to counts[vals]
    # Triton expects integer pointer; counts are int32
    for i in range(BLOCK):
        if mask[i]:
            tl.atomic_add(counts_ptr + vals[i], 1)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Compute offsets[1..E] = inclusive prefix sum of counts_ptr[0..E-1]
    # Initialize offsets[0] on host; kernel starts at 1
    sum_val = 0
    for i in range(0, E):
        sum_val += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, sum_val)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: out holds sorted positions, starts holds exclusive prefix per expert
    # Iterate positions in ascending order
    for pos in range(0, N):
        idv = tl.load(x_ptr + pos)
        # Write current pos into out at starts[idv]
        tl.store(out_ptr + tl.load(starts_ptr + idv), pos)
        # Advance starts[idv] by 1
        tl.atomic_add(starts_ptr + idv, 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Use the provided tensor exactly; do not generate or alter it.
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing (metadata, no torch ops)
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
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums: starts[e] = offsets[e]
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
