import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_atomic(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Parallel atomic histogram:
    - Launch with grid=(M,) where M is arbitrary number of programs.
    - Each program iterates over the flat array in chunks of size BLOCK, loads values,
      compares to all expert ids [0..num_experts-1], computes matches per expert, and
      atomically adds to counts_ptr[expert].
    - This guarantees full coverage of all elements regardless of grid size.
    """
    pid = tl.program_id(0)
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < N
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)

        # For each expert e, count matches in this chunk and atomically add to global counts[e]
        for e in range(num_experts):
            matches = (vals == e) & mask
            matches_i32 = matches.to(tl.int32)
            cnt_block = tl.sum(matches_i32, axis=0)
            tl.atomic_add(counts_ptr + e, cnt_block)

        offset += BLOCK


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts into offsets_ptr[0..N_bins-1], where
    offsets[i] = sum_{k < i} counts[k]. We write offsets[0] = 0; offsets[1..N_bins] = prefix.
    This kernel is simple and O(N_bins^2), acceptable for N_bins=256.
    """
    # Initialize offsets[0] = 0
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    # For each i from 1 to N_bins-1, offsets[i] = sum_{k=0..i-1} counts[k]
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version of the original Model.
    - Sorting is done via torch.sort (PyTorch) to maintain correctness and performance.
    - Histogram (counts per expert) is computed via a Triton parallel atomic kernel.
    - Exclusive prefix sum to produce expert_offsets is computed via Triton.
    """
    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        total = flat.numel()
        device = flat.device

        # 1) Sorting using PyTorch (stable=True to match original behavior)
        sorted_token_indices, _ = torch.sort(flat, dim=0, stable=True)

        # 2) Triton histogram (counts per expert) using parallel atomic adds


def run(*args):
    return ModelNew()(*args)
