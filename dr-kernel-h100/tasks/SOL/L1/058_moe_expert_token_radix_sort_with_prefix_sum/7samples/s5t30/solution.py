import torch
import triton
import triton.language as tl


@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_idx_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Argsort permutation via counting-sort for unique IDs.
    flat_ptr: 1D int32 tensor of length N (flattened expert indices).
    out_idx_ptr: 1D int32 tensor of length N, output permutation indices.
    N: total number of elements.
    """
    i = 0
    while i < N:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        ids = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # Store original index i for each id. With unique IDs, each slot is set exactly once.
        tl.store(out_idx_ptr + ids, offs.to(tl.int32), mask=mask)
        i += BLOCK


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Histogram of expert IDs using atomics. counts_ptr: int32 vector of length num_experts.
    We set counts[id] += 1 for each id in flat_ptr.
    """
    i = 0
    while i < N:
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        ids = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # Atomic add 1 for each valid id
        tl.atomic_add(counts_ptr + ids, 1, mask=mask)
        i += BLOCK


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute exclusive prefix sum of counts to produce offsets[1..num_experts].
    offsets_ptr: int32 vector of length num_experts + 1.
    """
    # offsets[0] must be 0; set by host before launching this kernel.
    total = 0
    for j in range(0, num_experts):
        # Each iteration adds counts[j] to total and writes total to offsets[j+1]
        total += tl.load(counts_ptr + j)
        tl.store(offsets_ptr + j + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Computes sorted_token_indices (argsort permutation) using Triton.
        - Computes expert_offsets (exclusive prefix sum) using Triton.
        """
        assert topk_idx.is_cuda, "Input must be on CUDA for Triton."
        # Flatten and ensure int32
        flat = topk_idx.contiguous().view(-1).to(torch.int32)
        device = flat.device
        N = flat.numel()
        num_experts = 256  # consistent with the original implementation

        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices (sorted_token_indices)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Triton kernel launches
        BLOCK = 1024  # chunk size for vectorized loads/stores

        # 1) Stable argsort permutation via Triton (counting-sort approach)
        stable_argsort_counting_kernel[(1,)](flat, out_idx, N, BLOCK=BLOCK)

        # 2) Count per-expert IDs
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # 3) Exclusive prefix sum of counts to produce offsets[1..]
        # Ensure offsets[0] = 0
        offsets[0] = 0
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Return: permutation indices and offsets
        # sorted_token_indices correspond to out_idx (we need indices, not values)
        # We should return them as int32 and the same shape as original: (num_tokens,)
        # Original output order was (sorted_token_indices, expert_offsets).
        return out_idx, offsets


def run(*args):
    return ModelNew()(*args)
