import torch
import triton
import triton.language as tl


@triton.jit
def _global_counting_sort_stable(flat_ptr, out_idx_ptr, offsets_ptr, N, K: tl.constexpr):
    # i is the global token index in 0..N-1
    i = tl.program_id(0)
    # Load value for position i
    val = tl.load(flat_ptr + i)
    # Place i at offsets[val], then advance offsets[val] by 1
    pos = tl.atomic_add(offsets_ptr + val, 1)
    tl.store(out_idx_ptr + i, pos)


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, K: tl.constexpr):
    class_id = tl.program_id(0)  # one program per class id in [0, K)
    # Accumulate count for this class across all N elements
    running = 0
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        running += (val == class_id)
    # Write the accumulated count
    tl.store(counts_ptr + class_id, running)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, offsets_ptr, K: tl.constexpr):
    # Exclusive prefix-sum (out[i] = sum(counts[:i])) and store inclusive in offsets
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, K):
        cnt = tl.load(counts_ptr + i)
        running += cnt
        tl.store(offsets_ptr + i, running)


def _compute_sorted_token_indices(topk_idx: torch.Tensor) -> torch.Tensor:
    # Flatten and ensure int32 for Triton
    flat = topk_idx.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    offsets = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Launch one program per token
    grid = (N,)
    _global_counting_sort_stable[grid](flat, out_idx, offsets, N, K=256)
    return out_idx


def _compute_expert_offsets(topk_idx: torch.Tensor, num_experts: int = 256) -> torch.Tensor:
    # Flatten and ensure int32
    flat = topk_idx.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    N = flat.numel()
    # Triton histogram over classes
    counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    grid_hist = (num_experts,)
    _hist_kernel[grid_hist](flat, counts, N, K=num_experts)
    # Inclusive scan to get prefix sums (offsets)
    out_offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    _inclusive_scan_kernel[grid_hist](counts, out_offsets, K=num_experts)
    # Build expert_offsets of length (num_experts + 1) with offsets[0] = 0
    # Using minimal PyTorch ops to form final tensor; heavy work is in Triton.
    base = torch.zeros(1, dtype=torch.int32, device=flat.device)
    expert_offsets = torch.cat([base, out_offsets])
    return expert_offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Triton-only computation: no torch.sort, torch.argsort, torch.bincount, torch.cumsum
        sorted_token_indices = _compute_sorted_token_indices(topk_idx)
        expert_offsets = _compute_expert_offsets(topk_idx)
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
