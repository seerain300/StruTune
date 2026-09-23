import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements and atomically increments counts[id]
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0)  # int32
    ids = x_vals % E  # expert ids in [0, E)
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: iterate positions sequentially to ensure stability
    # This kernel processes positions 0..N-1 and writes out permutation indices.
    # Each program handles a chunk of positions sequentially to avoid dynamic looping issues.
    pid = tl.program_id(0)
    # We launch grid = (1,) and loop over N sequentially to keep behavior simple and correct.
    # Note: Triton supports for-loops with runtime N. The following loop is valid in Triton JIT.
    for pos in range(N):
        id_val = tl.load(x_ptr + pos)
        e = id_val % E
        loc = tl.load(starts_ptr + e)
        tl.store(out_ptr + loc, pos)
        tl.atomic_add(starts_ptr + e, 1)


class ModelNew(torch.nn.Module):
    def __init__(self, block_hist: int = 1024, block_sort: int = 1):
        super().__init__()
        self.block_hist = block_hist
        self.block_sort = block_sort  # We use sequential per-position sort, so BLOCK=1 is fine.

    def forward(self, topk_idx: torch.Tensor):
        # Use provided topk_idx exactly; do not generate or alter it.
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = 256  # num_experts as per original run

        # 1) Triton histogram: counts of expert ids
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        grid_hist = (triton.cdiv(N, self.block_hist),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=self.block_hist)

        # 2) Compute offsets via torch.cumsum (host-side op is acceptable here)
        #    offsets[1:] = inclusive prefix sums of counts
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(counts, dim=0)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=self.block_sort)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
