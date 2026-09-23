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

    # Initialize current prefix sum for this tile
    prefix = tl.zeros([BLOCK], dtype=tl.int32)
    # Iterate over elements in this tile to compute inclusive scan
    # For each element i in [start, start+BLOCK-1], compute offsets[i+1] = offsets[i] + counts[i]
    for i in range(0, BLOCK):
        idx = start + i
        valid = idx < E
        # Read previous offset (start from 0)
        prev = tl.load(offsets_ptr + idx, mask=valid, other=0)
        # Read current count
        cnt = tl.load(counts_ptr + idx, mask=valid, other=0)
        # Update prefix (carry) and store next offset
        prefix = prefix + cnt
        tl.store(offsets_ptr + idx + 1, prefix, mask=valid)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: write out[starts[id]] = pos, then starts[id] += 1.
    # Iterate positions in increasing order to preserve stability for ties.
    for pos in range(0, N):
        id = tl.load(x_ptr + pos)
        id = id.to(tl.int32)
        start = tl.load(starts_ptr + id)
        tl.store(out_ptr + start, pos)
        # Increment starts for this expert
        tl.atomic_add(starts_ptr + id, 1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256  # fixed per original run

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch, seq_len, num_experts_per_tok), int32 on CUDA
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 2048
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0  # inclusive prefix sum starts at 0
        BLOCK_SCAN = 256
        grid_scan = (triton.cdiv(E, BLOCK_SCAN),)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
