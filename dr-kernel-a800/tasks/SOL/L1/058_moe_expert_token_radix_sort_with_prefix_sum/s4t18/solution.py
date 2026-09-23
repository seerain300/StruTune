import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    @triton.jit
    def _histogram_atomic_kernel(flat_ptr, N, counts_ptr, BLOCK: tl.constexpr):
        # Each program processes a chunk of BLOCK elements and performs per-element atomic_add to counts.
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N

        idx = tl.load(flat_ptr + offsets, mask=mask, other=0)
        idx32 = idx.to(tl.int32)
        # Atomic add 1 for each valid index
        tl.atomic_add(counts_ptr + idx32, 1, mask=mask)

    @triton.jit
    def _prefix_inclusive_scan_experts(counts_ptr, offsets_ptr, num_experts, BLOCK: tl.constexpr):
        # Compute inclusive prefix sum of counts into offsets[1:], with offsets[0]=0.
        # This kernel iterates over the counts array in chunks and writes the inclusive sums.
        # Initialize offsets[0] = 0
        tl.store(offsets_ptr + 0, 0)
        carry = 0
        # Iterate through experts
        for i in range(0, num_experts):
            val = tl.load(counts_ptr + i)
            carry += val
            tl.store(offsets_ptr + i + 1, carry)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters

    def forward(self, *args):
        # Expect a single tensor: topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]

        # Ensure CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Triton histogram counts per expert id
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel: one program per chunk of BLOCK elements
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](flat, N, counts, BLOCK=BLOCK, num_warps=4)

        # Triton inclusive prefix sum to form expert_offsets (num_experts + 1)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        # Run prefix scan kernel (grid=1, process all experts in a loop)
        _prefix_inclusive_scan_experts[(1,)](counts, expert_offsets, num_experts, BLOCK=num_experts, num_warps=1)

        # Sort flattened indices in PyTorch (values-only, not tied to num_experts); keep stable
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
