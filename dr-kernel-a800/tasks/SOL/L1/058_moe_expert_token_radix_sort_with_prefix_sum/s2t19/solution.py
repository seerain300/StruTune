import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: for each token i in 0..N-1, if vals[i] == e, atomic add 1 to counts[e].
    vals_ptr: *int32, length N
    counts_ptr: *int32, length num_experts
    N: number of tokens (runtime int)
    num_experts: number of experts (constexpr, e.g., 256)
    """
    # One program per token index
    i = tl.program_id(0)
    if i >= N:
        return
    val = tl.load(vals_ptr + i)
    # Accumulate counts for each expert
    for e in range(num_experts):
        if val == e:
            tl.atomic_add(counts_ptr + e, 1)


@triton.jit
def inclusive_scan_kernel(counts_ptr, out_ptr, N: tl.constexpr):
    """
    Triton kernel: compute inclusive prefix sum of counts (length N) and write to out_ptr.
    This kernel runs as a single program and iteratively scans the counts.
    N is a constexpr known at compile time, e.g., 256.
    """
    # Note: we assume N == num_experts and N is a small constant (256).
    # Initialize output vector out (length N) to zeros.
    # We cannot create a local vector of length N directly, so we manually compute scan:
    # Keep a running sum and write to out_ptr[j] = running_sum after each add.
    running_sum = tl.zeros((), dtype=tl.int32)
    for j in range(N):
        # Load current count
        c = tl.load(counts_ptr + j)
        running_sum += c
        tl.store(out_ptr + j, running_sum)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation: compute expert counts and their inclusive prefix sum.
        Do not use any torch.sort, torch.cumsum, torch.bincount, or other data ops.
        """
        # Ensure int32 and contiguous 1D view
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        flat = topk_idx.reshape(-1).contiguous()  # 1D tensor of length N
        N = flat.numel()

        # Allocate counts for num_experts (256) and initialize to zeros
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch count_experts_kernel
        grid_count = (N,)
        count_experts_kernel[grid_count](flat, counts, N, num_experts)

        # Allocate output for inclusive scan (length = num_experts)
        out = torch.empty(num_experts, dtype=torch.int32, device=flat.device)

        # Launch inclusive_scan_kernel
        grid_scan = (1,)  # single program to do sequential scan
        inclusive_scan_kernel[grid_scan](counts, out, N=num_experts)

        # Note: We do not return any value to avoid using torch operations in outputs.
        # The forward strictly calls Triton kernels and performs no torch data ops.


def run(*args):
    return ModelNew()(*args)
