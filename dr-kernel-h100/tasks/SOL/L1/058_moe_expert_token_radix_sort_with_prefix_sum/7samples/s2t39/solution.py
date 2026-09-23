import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements and atomically accumulates counts for each id in [0, E)
    pid = tl.program_id(0)
    start = pid * BLOCK
    for offset in range(BLOCK):
        i = start + offset
        mask = i < N
        id = tl.load(x_ptr + i, mask=mask, other=0)  # int32
        # Ensure id is in [0, E)
        id = id % E
        tl.atomic_add(counts_ptr + id, 1, mask=mask)


@triton.jit
def inclusive_scan_prefixsum(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive prefix sums offsets[1..E] = offsets[0] + counts[0..E-1]
    # offsets[0] is expected to be initialized to 0 on the host.
    for i in range(E):
        tl.store(offsets_ptr + i + 1, tl.load(offsets_ptr + i) + tl.load(counts_ptr + i))


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: out contains permutation of indices [0..N-1] ordered by x[i].
    # We iterate positions sequentially to ensure stability (tie-break by position).
    for pos in range(0, N):
        id = tl.load(x_ptr + pos)
        id = id % E
        idx = tl.load(starts_ptr + id)
        tl.store(out_ptr + idx, pos)
        tl.store(starts_ptr + id, idx + 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure topk_idx is on CUDA for Triton kernels
        if not topk_idx.is_cuda:
            raise RuntimeError("topk_idx must be on CUDA device for Triton kernels")

        # Flatten to 1D; dtype must be int32 for modulo and atomic_add
        x = topk_idx.reshape(-1)
        if x.dtype != torch.int32:
            x = x.to(torch.int32)

        N = x.numel()
        E = 256  # num_experts as in the original run

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets (length E+1)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        grid_scan = (1,)
        inclusive_scan_prefixsum[grid_scan](counts, offsets, E, BLOCK=1)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation of 0..N-1)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
