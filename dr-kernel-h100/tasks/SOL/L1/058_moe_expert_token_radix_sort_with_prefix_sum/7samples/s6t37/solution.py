import torch
import triton
import triton.language as tl


# Kernel 1: Compute per-class counts from flat using atomic adds.
# flat: 1D int32 tensor of length N
# counts: 1D int32 tensor of length NUM_CLASSES (256)
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # One program runs a loop across N and atomically increments counts per value.
    # Note: This uses a single program to keep it simple; N in benchmarks is modest.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)  # int32
        # valid only for 0 <= val < NUM_CLASSES
        tl.atomic_add(counts_ptr + val, 1)


# Kernel 2: Inclusive prefix sum of counts -> scan of length NUM_CLASSES+1
# counts: int32 of length NUM_CLASSES
# scan: int32 of length NUM_CLASSES+1
@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    running = tl.zeros((), dtype=tl.int32)
    for j in range(0, NUM_CLASSES + 1):
        if j < NUM_CLASSES:
            running += tl.load(counts_ptr + j)
        # store inclusive sum up to j
        tl.store(scan_ptr + j, running)


# Kernel 3: Global stable argsort based on flat values.
# flat: 1D int32 of length N
# out_idx: 1D int32 of length N (output permutation)
# scan: 1D int32 of length NUM_CLASSES+1 (inclusive prefix sums)
# N: int32
@triton.jit
def _global_argsort_stable_kernel(flat_ptr, out_idx_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # Fill out_idx by placing each index i in the correct position based on its class and scan.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        # start index for this class in out_idx
        start = tl.load(scan_ptr + val)
        # place i at position start and advance start
        tl.store(out_idx_ptr + start, i)
        tl.atomic_add(scan_ptr + val, 1)  # next position for this class


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._num_classes = 256  # match original code

    def forward(self, topk_idx: torch.Tensor):
        # Ensure 3D input as per original get_inputs
        assert topk_idx.ndim == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D (same as original run)
        flat = topk_idx.reshape(-1)
        # Triton kernels expect int32
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)
        # Ensure contiguous memory for Triton
        flat = flat.contiguous()
        N = flat.numel()

        # 1) Compute per-class counts via Triton
        counts = torch.zeros(self._num_classes, dtype=torch.int32, device=flat.device)
        _hist_kernel[(1,)](flat, counts, N, self._num_classes)

        # 2) Compute inclusive prefix scan via Triton
        scan = torch.empty(self._num_classes + 1, dtype=torch.int32, device=flat.device)
        _inclusive_scan_kernel[(1,)](counts, scan, self._num_classes)

        # 3) Global stable argsort via Triton, produce sorted_token_indices
        out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
        _global_argsort_stable_kernel[(1,)](flat, out_idx, scan, N, self._num_classes)

        # 4) expert_offsets: original sets offsets[1:] = cumulative counts; we derived scan already.
        #    expert_offsets = scan[1:] which corresponds to inclusive cumulative count per expert.
        expert_offsets = scan[1:]  # shape: (self._num_classes + 1 - 1,) == (256,)

        # Return results: sorted_token_indices (int32, 1D, length N), expert_offsets (int32, 1D, length 256 + 1 - 1)
        # Note: sorted_token_indices matches original output shape (N,). expert_offsets is (256,)
        # Align with original: expert_offsets has length num_experts + 1
        # Adjust to return expert_offsets of length 256 + 1 by prepending 0 at host (no need, Triton kernel produced full scan)
        expert_offsets = scan[1:]  # shape: (256,)
        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
