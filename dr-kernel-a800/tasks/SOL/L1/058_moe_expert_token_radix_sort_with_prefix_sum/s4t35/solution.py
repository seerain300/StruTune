import torch
import triton
import triton.language as tl


@triton.jit
def _hist_atomic_kernel(topk_ptr, counts_ptr, N, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton histogram kernel using per-element atomic_add:
    - Each program processes BLOCK_SIZE elements.
    - For each valid element, atomically increments the corresponding bin.
    - Reduces kernel complexity and minimizes local work.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load a block of top-k indices (int32)
    x = tl.load(topk_ptr + offs, mask=mask, other=0)

    # For each lane, if valid, atomically increment the corresponding bin
    for j in range(BLOCK_SIZE):
        val = x[j]  # scalar int32, guaranteed in [0, NUM_EXPERTS-1] per original code
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton prefix-sum kernel:
    - Computes inclusive prefix sums over NUM_EXPERTS and writes to offsets[1..].
    - offsets[0] is initialized to 0 on host.
    """
    # Single program computes the entire prefix sum (NUM_EXPERTS is small, e.g., 256)
    prefix = tl.zeros((), dtype=tl.int32)
    for j in range(NUM_EXPERTS):
        val = tl.load(counts_ptr + j)
        prefix += val
        tl.store(offsets_ptr + j + 1, prefix)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized computation:
        - Flatten topk_idx to 1D, ensure int32 and contiguous
        - Compute per-expert counts using Triton atomic histogram kernel
        - Compute expert_offsets via Triton prefix-sum kernel
        Return sorted_token_indices (torch.argsort stable) and expert_offsets.
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Flatten and ensure contiguous int32
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()
        num_experts = 256  # fixed in original code

        # Allocate counts and offsets
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        offsets[0] = 0  # initialize first offset to 0

        # Launch histogram kernel: 1D grid over blocks of size BLOCK_SIZE
        BLOCK_SIZE = 2048  # tuned for throughput; adjust if needed
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _hist_atomic_kernel[grid](flat, counts, N, NUM_EXPERTS=num_experts, BLOCK_SIZE=BLOCK_SIZE)

        # Launch prefix-sum kernel: single program computes full array
        _prefix_sum_kernel[(1,)](counts, offsets, NUM_EXPERTS=num_experts, BLOCK_SIZE=256)

        # sorted_token_indices: use PyTorch's stable sort on the original flattened tensor
        sorted_token_indices = torch.argsort(flat, stable=True)

        # Return results: keep dtypes and shapes as in original
        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
