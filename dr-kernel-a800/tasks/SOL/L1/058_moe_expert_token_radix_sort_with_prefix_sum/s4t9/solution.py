import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic_kernel(x_ptr, out_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Simple and robust histogram: each program processes BLOCK elements, loads x[i], casts to int32,
    then atomically adds 1 to out[x[i]] for valid indices in [0, num_experts-1].
    out_ptr is a 1D int32 array of length num_experts.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values (assumed to be integer indices)
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x = x.to(tl.int32)

    # Only update for indices in valid range
    valid = (x >= 0) & (x < num_experts) & mask

    # Atomic add 1 for each valid index
    # We use a vectorized atomic_add: for each lane with valid, increment the corresponding bin.
    # Note: out_ptr is contiguous 1D int32.
    for i in tl.static_range(0, BLOCK):
        # Triton allows scalar indexing in for-loops; for each i where valid[i] is True, do atomic add.
        if valid[i]:
            tl.atomic_add(out_ptr + x[i], 1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; Triton kernels will be launched in forward.

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only data-dependent computation:
        - Compute histogram of expert IDs in Triton.
        - Sort flattened indices stably using torch (PyTorch).
        - Compute expert_offsets via torch.cumsum on GPU.
        """
        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Triton histogram: counts per expert id
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch kernel: one program per BLOCK elements
        BLOCK = 1024  # tile size; N is typically small in provided workloads
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](
            flat, counts, N,
            num_experts=num_experts,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Compute expert offsets: inclusive prefix sum of counts (on GPU)
        # expert_offsets[i+1] = sum of counts[0..i], with expert_offsets[0] = 0
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        running = 0
        for i in range(num_experts):
            running += counts[i]
            expert_offsets[i + 1] = running

        # Stable sort of flattened indices using PyTorch (robust and matches original)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
