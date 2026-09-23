import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Compute histogram per expert id without torch ops
    pid = tl.program_id(0)
    start = pid * BLOCK
    # Vector of indices for this program
    idx = start + tl.arange(0, BLOCK)
    mask = idx < N
    # Load x at valid positions; masked invalid positions won't be used
    vals = tl.load(x_ptr + idx, mask=mask, other=0)
    # Atomically add 1 to counts[vals] for each valid element
    # Note: vals are int32 (as per input), counts are int32
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Compute inclusive prefix sums: offsets[i+1] = sum_{j=0..i} counts[j]
    # We assume counts_ptr[0..E-1] is already filled.
    # We write offsets[1..E] in-kernel; offsets[0] is set by host to 0.
    running = tl.zeros((), dtype=tl.int32)
    # Use a simple scalar loop across E; E is passed as runtime int
    i = 0
    while i < E:
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(offsets_ptr + i + 1, running)
        i += 1


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort producing permutation: out[pos] = sorted position
    # We iterate positions sequentially to ensure stability.
    pos = 0
    while pos < N:
        val = tl.load(x_ptr + pos)
        start = tl.load(starts_ptr + val)
        tl.store(out_ptr + start, pos)
        # advance starts[val] by 1
        current = tl.load(starts_ptr + val)
        tl.store(starts_ptr + val, current + 1)
        pos += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256  # same as the original run

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is CUDA and int32 as required
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing (no torch ops)
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
