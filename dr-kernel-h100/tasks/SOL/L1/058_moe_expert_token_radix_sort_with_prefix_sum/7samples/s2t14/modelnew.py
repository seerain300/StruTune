import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Compute per-expert histogram without atomics: each program processes BLOCK elements
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0)

    for id in range(0, E):
        eq = (x_vals == id) & mask
        cnt = tl.sum(eq.to(tl.int32), axis=0)
        tl.store(counts_ptr + id, cnt)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Compute offsets[1..E] as inclusive prefix sums of counts; offsets[0] is set by host to 0
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, E):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, running)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    # Fill permutation out stably: for each position pos, write out[starts[e]] = pos, then starts[e] += 1
    pos = 0
    while pos < N:
        id = tl.load(x_ptr + pos)
        start = tl.load(starts_ptr + id)
        tl.store(out_ptr + start, pos)
        tl.store(starts_ptr + id, start + 1)
        pos += 1


class ModelNew(torch.nn.Module):
    def __init__(self, axes_and_scalars: dict):
        super().__init__()
        # Store provided scalars; original code uses fixed num_experts=256
        self.num_experts = int(axes_and_scalars["num_experts"])

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx is expected to be provided by get_inputs; do not use any torch ops on it other than reshape
        x = topk_idx.reshape(-1)  # int32 on device

        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.empty(E, dtype=torch.int32, device=x.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK)

        # 2) Compute expert offsets via inclusive prefix sum in Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets.fill_(0)  # offsets[0] = 0
        grid_scan = (E,)
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums; starts[e] = offsets[e]
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E)

        # Return sorted_token_indices (int32 permutation) and expert_offsets
        return out, offsets