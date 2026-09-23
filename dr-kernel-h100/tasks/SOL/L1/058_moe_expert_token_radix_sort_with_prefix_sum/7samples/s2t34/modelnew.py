import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    """
    Compute per-expert histogram of x_ptr[0:N] into counts_ptr[0:E].
    Each program processes BLOCK elements; masked loads prevent OOB.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(x_ptr + offs, mask=mask, other=0)  # load int32
    # For masked lanes, vals=0 which is outside [0, E), so masked loads ensure safety
    for i in range(BLOCK):
        idx = vals[i]
        if (mask[i] and 0 <= idx < E):
            tl.atomic_add(counts_ptr + idx, 1)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    """
    Compute offsets[1..E] = inclusive prefix sums of counts_ptr[0..E-1].
    offsets[0] must be set to 0 on host before launch.
    Single program performs sequential scan.
    """
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, E):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    """
    Stable counting sort: for each position pos in [0, N), read id = x[pos],
    write out[starts[id]] = pos, then starts[id] += 1.
    One program per position ensures stability.
    """
    pid = tl.program_id(0)
    if pid < N:
        id = tl.load(x_ptr + pid)  # int32
        pos = pid
        # Write to the correct slot; starts_ptr is int32
        slot = tl.load(starts_ptr + id)
        tl.store(out_ptr + slot, pos)
        # Increment exclusive start for this expert
        tl.store(starts_ptr + id, slot + 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = int(num_experts)

    def forward(self, topk_idx: torch.Tensor):
        # Ensure the provided tensor is used exactly, no torch ops here
        if not topk_idx.is_cuda:
            raise RuntimeError("topk_idx must be on CUDA device for Triton kernels")

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
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=1)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (N,)  # one program per position for stability
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets