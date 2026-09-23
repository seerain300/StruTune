import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements of x
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load values; missing lanes get 0 (will not be counted due to mask)
    vals = tl.load(x_ptr + offs, mask=mask, other=0)
    # For each possible expert id, accumulate counts via atomic_add
    for v in range(E):
        m = (vals == v) & mask
        # Atomic add 1 for each matching lane
        # Note: Triton supports atomic_add on int32 tensors
        counts_ptr = counts_ptr + v
        tl.atomic_add(counts_ptr, tl.sum(m.to(tl.int32)))


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute offsets[i+1] = offsets[i] + counts[i], with offsets[0] = 0
    for i in range(E):
        # Each program handles one index i
        cur = tl.load(counts_ptr + i)
        if i == 0:
            tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
        else:
            prev = tl.load(offsets_ptr + (i - 1))
            tl.store(offsets_ptr + i, prev + cur)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable sort: out holds permutation of positions [0..N-1] by expert id
    for pos in range(N):
        id = tl.load(x_ptr + pos)
        dest = starts_ptr + id
        tl.store(out_ptr + tl.load(dest), pos)
        tl.atomic_add(dest, 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Use the provided topk_idx exactly (no torch ops)
        # Ensure dtype and device
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = 256  # num_experts as per original code

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton (one program per index)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        BLOCK_SCAN = 1  # per-index scan
        grid_scan = (E,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
