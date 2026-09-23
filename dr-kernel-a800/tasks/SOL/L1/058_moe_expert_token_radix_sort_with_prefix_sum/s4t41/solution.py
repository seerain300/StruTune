import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_experts_kernel(
    flat_ptr,           # *int32, flattened topk_idx
    counts_ptr,         # *int32, length = num_experts
    N,                  # int32, number of tokens
    num_experts: tl.constexpr,  # constexpr for indexing
    BLOCK_SIZE: tl.constexpr,   # constexpr block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load a block of token values
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32

    # For each element in the block, atomically increment counts[vals[i]]
    # Note: vals are assumed in [0, num_experts-1]; mask ensures we don't
    # touch out-of-range indices.
    for i in range(BLOCK_SIZE):
        idx = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + idx, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(
    counts_ptr,          # *int32, length = num_experts
    offsets_ptr,         # *int32, length = num_experts + 1
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single program kernel: scan counts in chunks, compute inclusive prefix sums,
    # and store to offsets. We do a sequential loop over chunks; num_experts is small.
    running = tl.zeros((), dtype=tl.int32)
    for start in range(0, num_experts, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_experts
        vals = tl.load(counts_ptr + offs, mask=mask, other=0)
        # Accumulate local running sums
        for i in range(BLOCK_SIZE):
            v = vals[i]
            if mask[i]:
                running += v
            # Store inclusive prefix sum at offsets[offs[i]+1]
            tl.store(offsets_ptr + (offs[i] + 1), running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Flatten topk_idx and use torch.sort for stable sorting of values.
        - Compute per-expert counts via Triton histogram kernel.
        - Compute expert_offsets via Triton inclusive prefix sum kernel.
        Returns:
          sorted_token_indices: int32 1D tensor of length N
          expert_offsets: int32 1D tensor of length num_experts+1
        """
        # Ensure CUDA and contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        topk_idx = topk_idx.contiguous()

        # Flatten the tensor
        flat = topk_idx.reshape(-1)  # shape: (N,)
        N = flat.numel()
        num_experts = 256  # matches original run function

        # Use torch.sort for correctness (values are sorted stably)
        _, sorted_token_indices = flat.sort(stable=True)

        # Allocate per-expert counts (int32)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch Triton histogram kernel
        # Choose a large BLOCK_SIZE to reduce grid and atomic pressure.
        BLOCK_SIZE_HIST = 4096
        grid = (triton.cdiv(N, BLOCK_SIZE_HIST),)
        _histogram_experts_kernel[grid](
            flat, counts, N,
            num_experts=num_experts,
            BLOCK_SIZE=BLOCK_SIZE_HIST,
            num_warps=4,
        )

        # Allocate expert_offsets and compute inclusive prefix sums via Triton
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

        # The prefix sum is over a small vector (num_experts). We can use a single program.
        _inclusive_prefix_sum_kernel[(1,)](
            counts, expert_offsets,
            num_experts=num_experts,
            BLOCK_SIZE=num_experts,
            num_warps=1,
        )

        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
