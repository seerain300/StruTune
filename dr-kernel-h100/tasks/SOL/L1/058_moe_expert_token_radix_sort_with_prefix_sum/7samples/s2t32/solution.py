import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E):
    """
    One-program-per-element histogram: reads x[i] and atomically increments counts[x[i]] if 0 <= x[i] < E.
    """
    pid = tl.program_id(0)
    if pid < N:
        val = tl.load(x_ptr + pid)
        # Only increment if val is a valid expert id
        if (val >= 0) & (val < E):
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    """
    Single-program inclusive prefix sum for counts_ptr[0:E] into offsets_ptr[1:E+1],
    with offsets_ptr[0] initialized to 0 on host.
    """
    total = 0
    for i in range(E):
        ci = tl.load(counts_ptr + i)
        total += ci
        tl.store(offsets_ptr + i + 1, total)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    """
    Stable counting sort: write position indices (0..N-1) into out at starts[x[pos]].
    After writing, increment starts[x[pos]].
    This preserves stability because positions are processed in increasing order.
    """
    for pos in range(N):
        id = tl.load(x_ptr + pos)
        idx = tl.load(starts_ptr + id)
        tl.store(out_ptr + idx, pos)
        tl.store(starts_ptr + id, idx + 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts  # 256 in the original code

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Flatten topk_idx
        - histogram_experts to counts
        - inclusive_scan_counts to expert_offsets
        - stable_counting_sort to produce sorted_token_indices (permutation)
        Returns: (sorted_token_indices: int32[N], expert_offsets: int32[E+1])
        """
        # Ensure topk_idx is on CUDA and int32, matching original input from get_inputs
        if not topk_idx.is_cuda:
            raise RuntimeError("topk_idx must be on CUDA device for Triton kernels")
        if topk_idx.dtype != torch.int32:
            raise RuntimeError("topk_idx must be of dtype torch.int32")

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        grid_hist = (N,)  # one program per element
        histogram_experts[grid_hist](x, counts, N, E)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        grid_scan = (1,)  # single program does the scan
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (N,)  # one program per position for stability
        stable_counting_sort[grid_sort](x, starts, out, N, E)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
