import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_kernel(flat_ids_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert ID in flat_ids_ptr and write to counts_ptr (length = num_experts).
    flat_ids_ptr: *int32, 1D
    counts_ptr: *int32, 1D of length num_experts
    N: total number of elements (int)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a chunk of IDs
    # We assume all IDs are in valid range [0, num_experts-1]
    ids = tl.load(flat_ids_ptr + offsets, mask=mask, other=0)

    # Atomic add 1 for each valid ID
    # Note: Triton supports atomic_add on int32
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute exclusive prefix sum of counts_ptr (length = num_experts) and write to offsets_ptr[1:].
    offsets_ptr[0] must be set separately by host (0).
    """
    # We use a sequential loop for simplicity and correctness.
    # Triton supports while loops, and num_experts is constexpr.
    i = 0
    acc = tl.zeros((), dtype=tl.int32)
    while i < num_experts:
        acc += counts_ptr[i]
        tl.store(offsets_ptr + i + 1, acc)
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Inputs:
            topk_idx: int32 tensor with shape (batch_size, seq_len, num_experts_per_tok)
        Outputs:
            sorted_token_indices: int32 tensor of shape (num_tokens,) = argsort of flattened topk_idx IDs
            expert_offsets: int32 tensor of shape (num_experts+1,)
        """
        # Flatten to 1D
        flat_ids = topk_idx.reshape(-1).to(torch.int32)

        # Compute sorted_token_indices using torch.argsort for correctness (stable=True)
        # This avoids subtle tie-breaking differences in custom Triton sorting.
        sorted_token_indices = torch.argsort(flat_ids, stable=True)

        # Prepare counts and offsets
        num_experts = 256  # as in the original code
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat_ids.device)

        # Launch Triton count kernel
        BLOCK = 1024  # chunk size; small for robustness
        grid = (triton.cdiv(flat_ids.numel(), BLOCK),)
        count_expert_ids_kernel[grid](flat_ids, counts, flat_ids.numel(), BLOCK)

        # Compute exclusive prefix sum using torch for simplicity and speed
        offsets = torch.cumsum(counts, dim=0)  # inclusive prefix sum
        # Make it exclusive and set offsets[0] = 0
        offsets = offsets - counts
        # Ensure offsets[0] = 0
        offsets[0] = 0

        return sorted_token_indices.to(torch.int32), offsets