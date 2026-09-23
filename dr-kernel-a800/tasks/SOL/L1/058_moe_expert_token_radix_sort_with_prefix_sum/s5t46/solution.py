import torch
import triton
import triton.language as tl


@triton.jit
def stable_permutation_kernel(
    flat_ptr,                 # *int32, flattened input of length M
    sorted_ptr,               # *int32, output permutation of length M
    M: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,      # number of output indices per program
):
    # Each program handles a block of output indices
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < M

    # Load values for these indices
    vals_j = tl.load(flat_ptr + offs, mask=mask, other=0)

    # Compute base_excl: inclusive prefix sum up to (key - 1)
    base_excl = tl.zeros([BLOCK], dtype=tl.int32)
    for k in range(NUM_EXPERTS):
        base = tl.sum((vals_j > k).to(tl.int32), axis=0)
        base_excl += base

    # Compute tie_count: for each j, count previous t < j with same key and vals_t < vals_j
    tie_count = tl.zeros([BLOCK], dtype=tl.int32)
    for t_off in range(0, M, BLOCK):
        t = t_off + tl.arange(0, BLOCK)
        mask_t = t < M
        vals_t = tl.load(flat_ptr + t, mask=mask_t, other=0)
        # For each j in this block, scan t block and count ties
        for jj in range(BLOCK):
            j_valid = (start + jj) < M
            j_val = vals_j[jj]
            # Count where t < (start + jj) and vals_t == j_val
            t_eq = (vals_t == j_val) & mask_t & (t < (start + jj))
            tie_count[jj] += tl.sum(t_eq.to(tl.int32))

    # Rank is base_excl + tie_count (stable ordering for equal keys by original index)
    rank = base_excl + tie_count
    tl.store(sorted_ptr + offs, rank, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA for Triton
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        # Flatten
        flat = topk_idx.reshape(-1)
        M = flat.numel()

        # Output tensor for sorted token indices
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)

        # Launch Triton kernel that computes stable permutation
        grid = (triton.cdiv(M, 1024),)
        stable_permutation_kernel[grid](flat, sorted_token_indices, M, self.num_experts, BLOCK=1024)

        # Return only sorted_token_indices (matches original run signature)
        return sorted_token_indices


def run(*args):
    return ModelNew()(*args)
