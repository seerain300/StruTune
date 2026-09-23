import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(
    x_ptr,               # *int32, flattened input
    counts_ptr,          # *int32, length num_experts
    n_elements: tl.constexpr,  # total number of elements in x
    num_experts: tl.constexpr, # number of distinct expert ids
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a chunk of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load values from x. For masked-out lanes, use 0 (won't contribute due to mask).
    vals = tl.load(x_ptr + offs, mask=mask, other=0)

    # For each value in this chunk, atomically add to counts[val].
    # Note: num_experts is a constexpr, so this loop is compiled with fixed bounds.
    for e in range(num_experts):
        # mask_e is True only for lanes where vals == e; masked atomic_add prevents OOB.
        mask_e = mask & (vals == e)
        tl.atomic_add(counts_ptr + e, tl.sum(mask_e.to(tl.int32)))


@triton.jit
def _inclusive_prefix_sum_kernel(
    counts_ptr,          # *int32, length num_experts
    offsets_ptr,         # *int32, length num_experts + 1
    num_experts: tl.constexpr
):
    # Single program instance computes prefix sum sequentially.
    running = 0
    # offsets_ptr[0] can be set by host; we write [1:] here.
    for e in range(num_experts):
        running += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e + 1, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Computes expert counts in Triton via atomic adds.
        - Computes expert offsets (cumulative sum) in Triton.
        - Uses PyTorch for stable sort to obtain sorted_token_indices (permutation).
        """
        # Ensure we're on CUDA and contiguous
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = topk_idx.contiguous()
        n = x.numel()
        num_experts = 256  # as in the original run function

        # 1) Triton histogram of expert IDs
        counts = torch.zeros(num_experts, dtype=torch.int32, device=x.device)
        # Choose a block size and grid
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](
            x, counts,
            n_elements=n,
            num_experts=num_experts,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4  # reasonable default
        )

        # 2) Triton inclusive prefix sum to produce expert offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=x.device)
        # We can leave offsets[0] = 0 and write offsets[1:] in the kernel.
        offsets[0] = 0
        _inclusive_prefix_sum_kernel[(1,)](
            counts, offsets,
            num_experts=num_experts,
            num_warps=1
        )

        # 3) Use PyTorch for stable sort to get sorted_token_indices (permutation of 0..N-1)
        #    Flatten to 1D for sorting; then reshape back as needed.
        flat = x.view(-1)
        # torch.sort default is stable for float/double; for integer keys it's stable too.
        sorted_indices = torch.sort(flat, stable=True)[1]  # indices that would sort flat

        # Return as original API: sorted_token_indices (int64), expert_offsets (int32)
        # The original returns int32 for topk_idx, but sorted indices from torch are int64.
        # We keep int64 for sorted_token_indices to match PyTorch's default.
        return sorted_indices, offsets