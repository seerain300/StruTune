import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E):
    """
    Compute per-expert histogram of x_ptr[0:N] into counts_ptr[0:E].
    One program per element. Mask ensures no OOB access. Use masked load and atomic_add.
    """
    pid = tl.program_id(0)
    if pid < N:
        val = tl.load(x_ptr + pid)
        # Only increment valid expert ids in [0, E)
        if (val >= 0) & (val < E):
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    """
    Compute offsets[1..E] = inclusive prefix sums of counts_ptr[0..E-1],
    with offsets[0] already set to 0 on host.
    One program performing a sequential scan over E.
    """
    # Use program_id(0) as dummy; we will call with grid=(1,)
    # Note: Triton kernels without explicit grid parameters run with grid=(1,)
    total = 0
    for i in range(E):
        c = tl.load(counts_ptr + i)
        total += c
        tl.store(offsets_ptr + i + 1, total)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    """
    Stable counting sort: writes permutation out where out[pos] = pos.
    We use exclusive starts per expert: starts[e] = sum_{j < e} counts[j].
    For each pos in [0, N), read id = x[pos], then out[starts[id]] = pos, and starts[id] += 1.
    One program per position ensures stability.
    """
    pid = tl.program_id(0)
    if pid < N:
        id = tl.load(x_ptr + pid)
        # Ensure id is in range
        if (id >= 0) & (id < E):
            start = tl.load(starts_ptr + id)
            tl.store(out_ptr + start, pid)
            # Update exclusive start for this expert
            tl.atomic_add(starts_ptr + id, 1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256  # constant per the original code

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of:
        - Stable sort permutation of flattened topk_idx by expert id (counting sort).
        - Histogram of expert ids.
        - Inclusive prefix sums for expert offsets.

        Inputs:
          topk_idx: int32 tensor of shape (batch_size, seq_len, num_experts_per_tok), on CUDA.

        Outputs:
          - sorted_token_indices: int32 permutation of length N = batch_size*seq_len*num_experts_per_tok
          - expert_offsets: int32 tensor of length num_experts + 1 (inclusive prefix sums)
        """
        # Use provided topk_idx; do not create with torch ops
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

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
        grid_scan = (1,)  # single program performs sequential scan
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
