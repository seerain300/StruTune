import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr,
                              n_elements: tl.int32,
                              num_experts: tl.int32,
                              BLOCK_SIZE: tl.constexpr):
    """
    Count occurrences of each expert ID in flat_ptr into counts_ptr[0:num_experts].
    Uses one atomic_add per element.
    """
    # One program per BLOCK_SIZE chunk
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load flat values; ensure masked loads don't affect atomics
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Atomically add 1 to counts[vals] for valid elements
    # Note: we assume vals are in range [0, num_experts-1] as per original code.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr,
                                  n_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0:n_experts] into offsets_ptr[1:n_experts+1].
    offsets_ptr[0] should be initialized to 0; we leave it untouched and start from index 1.
    """
    # Single program instance performs the scan sequentially over n_experts.
    # It's fine because num_experts is small (256).
    running = tl.zeros((), dtype=tl.int32)  # scalar accumulator
    # Loop over i = 0..n_experts-1: offsets[i+1] = running += counts[i]
    # Note: Triton allows static loops if bounds are tl.constexpr; here n_experts is passed as int.
    # We use a Python for-loop inside the kernel; Triton supports loops with non-constexpr bounds.
    for i in range(n_experts):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + 1 + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (B, S, EPT), int32, CUDA tensor
        Returns:
          - sorted_token_indices: permutation of 0..N-1 that sorts flattened topk_idx ascending (stable)
          - expert_offsets: int32, length (num_experts + 1), inclusive prefix sums of counts per expert
        """
        # Ensure tensor is on CUDA device; Triton requires CUDA
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        device = topk_idx.device
        flat = topk_idx.reshape(-1).contiguous()  # int32, 1D
        n = flat.numel()
        num_experts = 256  # per problem setup; adjust if needed

        # 1) Compute sorted_token_indices using PyTorch's stable sort (robust and correct)
        sorted_token_indices = torch.sort(flat, stable=True)[1].to(torch.int32)

        # 2) Triton histogram: counts per expert
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Choose a reasonable block size; 1024 works well for typical N.
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](flat, counts, n, num_experts, BLOCK_SIZE=BLOCK_SIZE)

        # 3) Triton inclusive prefix sum to produce expert_offsets
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # Initialize expert_offsets[0] to 0; we will write [1:] in the kernel
        # But since we didn't zero explicitly, set it here:
        expert_offsets.fill_(0)
        _inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, n_experts=num_experts)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
