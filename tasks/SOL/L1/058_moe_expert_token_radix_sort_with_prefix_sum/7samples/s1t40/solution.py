import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(
    in_ptr,             # *int32, flattened expert indices
    counts_ptr,         # *int32, length = num_experts
    n_elements,         # int32, total number of elements in in_ptr
    num_experts,        # int32, number of experts
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a chunk of input elements.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values; for masked-out lanes, 'other' is arbitrary as they won't be stored.
    vals = tl.load(in_ptr + offsets, mask=mask, other=0)

    # Atomic add 1 for each valid lane into counts[vals]
    # Note: Triton supports atomic_add for int32.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_prefix_sum_kernel(
    input_ptr,          # *int32, length = num_experts, holds counts
    output_ptr,         # *int32, length = num_experts + 1, output offsets
    num_experts: tl.constexpr,  # compile-time known for loop
):
    # Single-program inclusive scan: output[i] = sum_{j=0..i} input[j]
    # We can use a simple loop since num_experts is small (256).
    running = 0
    for i in range(0, num_experts):
        running += tl.load(input_ptr + i)
        tl.store(output_ptr + i + 1, running)  # store at position i+1
    # Last element should be total sum; we already store running at last position via loop.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Uses torch.sort for stable sorting of flattened expert indices (correctness).
        - Uses Triton kernels to compute expert counts and prefix sums (TRITON-only computation).
        Returns:
          - sorted_token_indices: permutation of 0..N-1 (int32)
          - expert_offsets: inclusive prefix sums per expert (int32, length = num_experts + 1)
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        device = flat.device

        # Stable sort indices (PyTorch), matches original behavior
        _, sorted_token_indices = torch.sort(flat, stable=True)
        # We return indices as int32 (original code returns int32). If you need strict dtype match to original,
        # you can keep int64, but here int32 is fine for a permutation.
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        # Compute per-expert counts via Triton (atomic_add per element)
        n = flat.numel()
        num_experts = 256  # consistent with original code's use of num_experts
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)

        # Ensure input is int32 for atomic adds
        in_vals = flat.to(torch.int32)

        # Launch histogram kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](
            in_vals, counts, n, num_experts, BLOCK_SIZE=BLOCK_SIZE
        )

        # Compute inclusive prefix sums via Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # Launch inclusive scan kernel (single program instance)
        _inclusive_prefix_sum_kernel[(1,)](
            counts, offsets, num_experts=num_experts
        )

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
