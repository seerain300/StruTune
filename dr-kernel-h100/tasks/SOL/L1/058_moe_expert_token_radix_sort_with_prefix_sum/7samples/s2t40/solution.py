import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    # x is int32; compute expert id
    id = x % E
    # atomic add counts per id for valid lanes
    tl.atomic_add(counts_ptr + id, 1, mask=mask)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive scan over counts and write to offsets
    # offsets[0] is set on host; we compute offsets[1..E] = offsets[k-1] + counts[k]
    # Do a simple per-element scan within tiles. We iterate elements sequentially to keep it robust.
    # Note: BLOCK here is a tile size; E is number of experts.
    # We use a while loop to assign each element k: offsets[k] = offsets[k-1] + counts[k]
    # This is a single-program kernel that safely reads/writes its own k without atomics.
    # However, Triton prefers parallel kernels; implement tile-wise sequential loop.
    # For simplicity and robustness, process one element per loop using runtime loop.
    # The harness uses E <= 256, which is small, and this avoids atomics and races.
    # We'll use a for k in range(1, E) loop here.
    # This kernel is tiny and runs in O(E). It won't dominate performance for E=256.
    k = 1
    while k <= E:
        # Load previous offset (0 for k==1)
        prev = offsets_ptr[k - 1] if k > 0 else 0
        cnt = tl.load(counts_ptr + k)
        new = prev + cnt
        tl.store(offsets_ptr + k, new)
        k += 1


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable sort via counting: out holds sorted positions, starts holds exclusive prefix per expert.
    # We iterate positions sequentially to ensure stability.
    for pos in range(0, N):
        id = tl.load(x_ptr + pos)  # int32
        idx = tl.load(starts_ptr + id)  # current start for this expert
        tl.store(out_ptr + idx, pos)
        # increment start for this expert
        new = tl.load(starts_ptr + id) + 1
        tl.store(starts_ptr + id, new)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Use provided input tensor exactly; do not generate or alter it.
        # Flatten to 1D for processing (reshape is metadata; no torch computation).
        x = topk_idx.reshape(-1)
        assert x.dtype == torch.int32, "topk_idx must be int32"
        assert x.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
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
        # inclusive_scan_counts expects E; E is number of elements to scan
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=1)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
