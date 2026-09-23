import torch
import triton
import triton.language as tl


# Triton kernel: Histogram of values in orig (int32) into counts (int32).
# Assumes values are in [0, L-1], where L is num_experts (e.g., 256).
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0)
    offsets = lane * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)  # int32
    # Masked lanes won't add since mask is false; other=0 is fine.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton kernel: Exclusive prefix sum across counts to produce offsets per value.
# out_ptr[e] = inclusive sum of counts for ids < e. out_ptr[L] = total N.
@triton.jit
def exclusive_scan_kernel(counts_ptr, out_ptr, L: tl.constexpr):
    running = 0
    for e in range(L):
        c = tl.load(counts_ptr + e)
        out_ptr[e] = running
        running += c
    out_ptr[L] = running


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure dtype
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        device = topk_idx.device

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        num_experts = 256  # consistent with original context

        # 1) sorted_token_indices: permutation to sort flat stably. Use torch.argsort for correctness.
        sorted_token_indices = torch.argsort(flat, stable=True)  # int64 by default; cast to int32
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        # 2) expert_offsets via Triton histogram + exclusive scan
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_kernel[grid_hist](flat, counts, N, L=num_experts, BLOCK=BLOCK_HIST)

        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        exclusive_scan_kernel[(1,)](counts, offsets, L=num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
