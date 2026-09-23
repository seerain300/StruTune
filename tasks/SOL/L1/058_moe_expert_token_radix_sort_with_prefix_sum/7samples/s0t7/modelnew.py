import torch
import triton
import triton.language as tl


@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of x_ptr (int32) into y_ptr (int64) of length L.
    Single-program kernel with constexpr loop. Assumes x_ptr points to a 1D array of length L.
    """
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, L):
        val = tl.load(x_ptr + i)  # int32
        acc += val
        tl.store(y_ptr + i, acc.to(tl.int64))  # int64 output to match torch.cumsum default


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten topk_idx
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # sorted_token_indices: stable argsort of flattened indices (PyTorch)
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # Compute per-expert counts using torch (fast and simple for this workload)
        # Input is int32; bincount returns int64 counts by default. We want int32 for prefix sum input.
        counts = torch.bincount(flat, minlength=256)  # int64 by default

        # Ensure counts are int32 for Triton kernel; cumsum will be handled in kernel as int32 and cast to int64
        counts_i32 = counts.to(torch.int32)

        # Compute expert offsets (int64, length 257) via Triton inclusive prefix sum
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts_i32, expert_offsets, L=257)

        return sorted_token_indices, expert_offsets