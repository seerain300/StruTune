import torch
import triton
import triton.language as tl


@triton.jit
def stable_by_value_kernel(flat_ptr, out_idx_ptr, N, num_experts: tl.constexpr):
    """
    For each index i (0..N-1), read value v = flat[i] (int32), and place original index i at position v in out_idx (int64).
    This produces a stable ordering because we process i in increasing order; equal values place lower i first.
    """
    i = tl.program_id(axis=0)
    if i < N:
        val = tl.load(flat_ptr + i)  # int32
        tl.store(out_idx_ptr + val, tl.cast(i, tl.int64))


@triton.jit
def count_histogram_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Compute histogram of int32 vals (length N) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(vals_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    pid = tl.program_id(axis=0)
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Inputs: topk_idx of shape (B, S, EPT), int32, on CUDA.
        Outputs:
          sorted_token_indices: int64, shape (N,), sorted positions corresponding to flat.
          expert_offsets: int32, shape (num_experts + 1,), inclusive prefix sums of counts.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels"
        # Flatten
        flat = topk_idx.reshape(-1)  # int32 on CUDA
        N = flat.numel()
        num_experts = 256  # same as reference

        # 1) Stable by value: out_idx holds int64 original positions
        out_idx = torch.empty(N, device=flat.device, dtype=torch.int64)

        # Launch kernel: one program per element
        grid = (N,)
        stable_by_value_kernel[grid](flat, out_idx, N, num_experts=num_experts)

        # 2) Histogram in Triton
        counts = torch.zeros(num_experts, device=flat.device, dtype=torch.int32)
        grid_hist = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_hist](flat, counts, N, num_experts=num_experts)

        # 3) Prefix sum in Triton (int64 offsets)
        offsets_i64 = torch.empty(num_experts + 1, device=flat.device, dtype=torch.int64)
        offsets_i64[0] = 0
        prefix_sum_kernel[grid_hist](counts, offsets_i64, num_experts=num_experts)

        # 4) Return outputs: sorted_token_indices (int64) and expert_offsets (int32)
        return out_idx, offsets_i64.to(torch.int32)


def run(*args):
    return ModelNew()(*args)
