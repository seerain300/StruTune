import torch
import triton
import triton.language as tl


@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_idx_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Stable argsort permutation via counting-sort approach for unique IDs.
    For each original index i in [0, N), loads id = flat[i] and sets out_idx[id] = i.
    Assumes flat_ptr points to int32 values, out_idx_ptr to int32 values.
    """
    # Single program instance handles the whole array using chunks of BLOCK
    for start in range(0, N, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < N
        # Load IDs; 'other=0' for masked lanes
        ids = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Store out_idx[ids] = idx where idx < N
        tl.store(out_idx_ptr + ids, idx, mask=mask)


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Build histogram of expert IDs: counts[id] += 1 for each id in flat_ptr.
    Uses chunked iteration and atomic_add for safety across blocks.
    """
    for start in range(0, N, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < N
        ids = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Atomic add 1 for valid lanes
        tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute exclusive prefix sum of counts into offsets[1..].
    Assumes offsets_ptr length = num_experts + 1.
    offsets[0] must be set to 0 on host before launching this kernel.
    """
    # We initialize offsets[0] = 0 in host code.
    # Sequential loop over num_experts: offsets[j] = offsets[j-1] + counts[j-1]
    # Loop body executes per program instance with a small BLOCK; here we use scalar iteration.
    # Note: Triton supports scalar while loops; we iterate num_experts times.
    prev = 0
    j = 0
    while j < num_experts:
        prev = prev + tl.load(counts_ptr + j)
        tl.store(offsets_ptr + j + 1, prev)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Computes sorted_token_indices (permutation of indices that sorts expert IDs) using Triton.
        - Computes expert_offsets via Triton histogram + prefix sum.
        """
        # Ensure device is CUDA and dtype int32 for flat
        device = topk_idx.device
        num_experts = 256  # matches original setup

        # Flatten to 1D int32
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = flat.numel()

        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices (sorted_token_indices)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Triton kernel launches (no torch compute)
        BLOCK = 1024  # chunk size; small for robustness

        # 1) Stable argsort permutation via Triton (counting-sort approach)
        stable_argsort_counting_kernel[(1,)](flat, out_idx, N, BLOCK=BLOCK)

        # 2) Count per-expert IDs using Triton
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # 3) Exclusive prefix sum of counts to produce offsets[1..]
        offsets[0] = 0
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Return: permutation indices and offsets
        # sorted_token_indices correspond to out_idx (int32)
        return out_idx, offsets