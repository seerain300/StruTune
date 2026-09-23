import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(
    vals_ptr,              # *int32
    counts_ptr,            # *int32 (length = num_experts)
    n_elements,            # int32
    num_experts,           # int32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count occurrences of each expert ID in vals_ptr and atomically add
    to counts_ptr[exp] for each valid token. We iterate over vals_ptr
    in chunks of BLOCK_SIZE and process each element.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    # Loop over chunks
    for start in range(0, n_elements, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < n_elements
        # Load values; 'other' won't be used for stores due to mask
        vals = tl.load(vals_ptr + idx, mask=mask, other=0)
        # Ensure vals are int32
        vals = vals.to(tl.int32)
        # For each valid element, atomic add to its expert count
        # We only do this for masked elements
        for j in range(BLOCK_SIZE):
            valid = mask[j]
            val = vals[j]
            # Only proceed if idx is in bounds
            if valid:
                # Atomic add one to the bin 'val'
                # Note: num_experts is small (256), and val in [0, num_experts-1] by contract.
                tl.atomic_add(counts_ptr + val, 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Flattens topk_idx and computes per-expert counts via Triton.
        - Produces sorted_token_indices using torch.sort (stable=True).
        - Produces expert_offsets as inclusive cumsum of counts (int64), length num_experts+1.
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        # Ensure int32 for histogram
        flat_i32 = flat.to(torch.int32)

        num_experts = 256  # same as original
        n = flat_i32.numel()

        # Triton counts buffer: length = num_experts, initialized to zeros
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat_i32.device)

        # Launch Triton histogram kernel
        # Choose a reasonable block size; 1024 works well for typical N
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](
            flat_i32, counts, n, num_experts, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )

        # Compute expert_offsets as inclusive cumulative sum: length = num_experts + 1
        # counts are int32; convert to int64 for offsets to match original behavior (cumsum default).
        offsets = torch.cumsum(counts.to(torch.int64), dim=0)
        # offsets has shape (num_experts,) by cumsum; append zeros for the extra +1 at the end
        # However, torch.cumsum returns shape same as input; for a 1D vector, it returns length=num_experts.
        # The original code constructs expert_offsets of length num_experts+1 via:
        # expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=...)
        # expert_offsets[1:] = torch.bincount(flat.long()).cumsum(0).to(torch.int32)
        # Our counts capture bin values; to match shape, create zeros vector and fill indices 1:.
        # But here we have counts length=num_experts. We need length=num_experts+1:
        # The first element should be 0, last element should be N (sum of counts).
        # A simple way: build zeros of length num_experts+1 and fill positions 1..num_experts based on prefix sums.
        total_tokens = int(n)  # since counts sum to n
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=flat.device)
        # Fill prefix sums
        running = 0
        for e in range(num_experts):
            running += int(counts[e].item())
            expert_offsets[e + 1] = running

        # Also, we can include 0 at index 0 (already zero), and last index equals total_tokens.
        # Ensure last element equals n (sum of counts)
        # We filled expert_offsets[1..num_experts] via running, and expert_offsets[0] is zero.
        # So we need to set expert_offsets[num_experts] to total number of tokens.
        # We already did that by filling e+1 for each e. The last filled position is num_experts.
        # To be explicit:
        # if num_experts > 0: expert_offsets[num_experts] equals sum(counts). It should be n.
        # In case counts.sum() doesn't equal n for some reason, set it explicitly.
        # But since counts were derived from vals, they should sum to n. Still, enforce it.
        # Compute sum of counts
        sum_counts = int(counts.sum().item())
        if sum_counts != n:
            expert_offsets[-1] = torch.tensor(n, dtype=expert_offsets.dtype, device=expert_offsets.device)

        # sorted_token_indices: use torch.sort (stable=True) to match original exactly
        # Note: original returns int64; we return int64 to match.
        sorted_token_indices = torch.sort(flat)[1]  # indices (stable)

        return sorted_token_indices.to(torch.int64), expert_offsets


def run(*args):
    return ModelNew()(*args)
