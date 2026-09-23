import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(inp_ptr, N, out_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Triton histogram kernel:
    For each element in inp_ptr[0:N], atomically increment the corresponding bin in out_ptr[idx].
    Each program handles BLOCK elements; grid size is cdiv(N, BLOCK).
    out_ptr has length num_experts (int32 counts).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load indices; cast to int32 for atomic add
    vals = tl.load(inp_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # For each expert id, atomically add 1 for each matching value
    for e in range(num_experts):
        matches = vals == e  # boolean vector
        # Reduce to scalar count with mask
        partial = tl.zeros((), dtype=tl.int32)
        for j in range(BLOCK):
            valid = offsets[j] < N
            m_j = matches[j]
            partial += m_j.to(tl.int32) * valid.to(tl.int32)
        # Atomically add partial to bin e
        tl.atomic_add(out_ptr + e, partial)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; required by the evaluation harness (NewModule).

    def forward(self, *args):
        """
        Triton-optimized version:
        - Flattens topk_idx, computes histogram in Triton, then creates expert_offsets via torch.cumsum.
        - Keeps the stable sort in PyTorch as in the original code.
        """
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]

        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Triton counts per expert id
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Kernel launch configuration
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        _histogram_kernel[grid](flat, N, counts, num_experts=num_experts, BLOCK=BLOCK, num_warps=4)

        # Compute expert offsets: inclusive prefix sum (inclusive cumsum)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        running = 0
        for i in range(num_experts):
            running += counts[i]
            expert_offsets[i + 1] = running

        # Stable sort of flattened indices (same as original)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
