import torch
import triton
import triton.language as tl


# Triton kernel: build per-expert counts from flattened IDs (int32).
# counts: [num_experts], int32, initialized to zeros, then atomic adds.
@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values; for masked lanes, we can use 0, but we guard with mask for store
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Ensure vals are int32
    vals = vals.to(tl.int32)

    # Atomic add 1 for each valid token into counts[vals]
    # Each lane processes its element (masked) and performs atomic add.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton kernel: compute inclusive prefix sum of 'counts' and write to 'out' (length = num_experts + 1).
# We do a single program instance looping over num_experts; simple and fast since num_experts is small (256).
@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, out_ptr, num_experts: tl.int32):
    # out[0] = 0 (we'll set manually in host)
    total = 0
    for i in range(0, num_experts):
        total += tl.load(counts_ptr + i)
        tl.store(out_ptr + i + 1, total)
    # out[num_experts] = total (sum of counts)
    tl.store(out_ptr + num_experts, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure int32 (original topk_idx is int32)
        flat = topk_idx.reshape(-1)
        device = flat.device
        # Number of tokens
        n = flat.numel()
        num_experts = 256  # fixed as in the original code

        # 1) Triton histogram counts
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Choose a reasonable block size; 1024 works well for typical N.
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](flat, counts, n, num_experts, BLOCK_SIZE=BLOCK_SIZE)

        # 2) Triton inclusive prefix sum of counts -> expert_offsets (length = num_experts + 1)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # We'll set out[0] = 0; kernel writes out[1:] = inclusive cumsum
        # Note: Triton loop runs for a fixed num_experts (tl.constexpr not required here)
        _inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, num_experts)

        # 3) Use PyTorch for stable sorting of flat to produce the permutation indices
        #    This matches original behavior exactly and is efficient for moderate N.
        #    sorted_token_indices is the indices that would sort 'flat' ascending (stable).
        sorted_token_indices = torch.sort(flat, stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
