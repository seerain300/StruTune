import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: for each token i in [0, N), if vals[i] == e, increment counts[e] by 1.
    vals_ptr: *int32, length N
    counts_ptr: *int32, length num_experts
    """
    # Each program handles one token i
    i = tl.program_id(0)
    if i >= N:
        return
    val = tl.load(vals_ptr + i)  # int32
    # For all e in 0..num_experts-1, increment counts[e] if val == e
    for e in range(0, num_experts):
        # Condition (val == e) is scalar boolean; when true, contributes +1 via add.
        tl.store(counts_ptr + e, tl.load(counts_ptr + e) + (val == e))


@triton.jit
def inclusive_scan_kernel(counts_ptr, out_ptr, num_experts: tl.constexpr):
    """
    Triton kernel: compute inclusive prefix sum of counts_ptr (length num_experts)
    and write to out_ptr (length num_experts).
    Uses a single program to scan sequentially. num_experts is small (e.g., 256).
    """
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_experts):
        total += tl.load(counts_ptr + i)
        tl.store(out_ptr + i, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only forward:
        - Do NOT use torch.sort, torch.cumsum, torch.bincount, stable=True, etc.
        - Define and launch Triton kernels; do not return any outputs.
        """
        # Ensure flat 1D and contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device
        num_experts = 256  #


def run(*args):
    return ModelNew()(*args)
