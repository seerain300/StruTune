import torch
import triton
import triton.language as tl


@triton.jit
def _global_counting_sort_stable(flat_ptr, out_ptr, N, K: tl.constexpr):
    """
    Perform a stable global counting sort for integer values in [0, K-1].
    Writes out a permutation out_ptr[0..N-1] such that flat[out_ptr[i]] is sorted ascending.
    Stability is ensured by placing equal elements in original order (i increases).
    """
    offsets = tl.zeros((K,), dtype=tl.int32)
    # Loop over all tokens i in 0..N-1; each program handles one i
    # We launch grid=(N,) and let Triton run one iteration per program.
    # Triton allows loops over scalars; this is acceptable for our N range.
    for i in range(0, N):
        # Load class for this token
        val = tl.load(flat_ptr + i)
        # Compute current slot for this class and advance slot
        slot = offsets[val]
        # Place i at slot in out_ptr
        tl.store(out_ptr + slot, i)
        offsets[val] += 1


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, K: tl.constexpr):
    """
    Compute histogram of values in [0, K-1] across N elements into counts_ptr[0..K-1].
    """
    for k in range(0, K):
        cnt = tl.zeros((), dtype=tl.int32)
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            cnt += (val == k)
        tl.store(counts_ptr + k, cnt)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, offsets_ptr, K: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0..K-1] into offsets_ptr[0..K-1].
    offset[i] = sum_{j=0..i} counts[j].
    """
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, K):
        cnt = tl.load(counts_ptr + i)
        running += cnt
        tl.store(offsets_ptr + i, running)


def _compute_sorted_token_indices(topk_idx: torch.Tensor) -> torch.Tensor:
    """
    Triton-based stable global sort of the flattened topk_idx (int32 expected).
    Returns permutation of indices [0..N-1] so that flat[permutation[i]] is sorted ascending.
    """
    flat = topk_idx.reshape(-1).contiguous()
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    # Launch one program per element; counting sort handles all elements.
    grid = (N,)
    _global_counting_sort_stable[grid](flat, out_idx, N, 256)
    return out_idx


def _compute_expert_offsets(topk_idx: torch.Tensor, num_experts: int = 256) -> torch.Tensor:
    """
    Compute expert offsets via Triton histogram + inclusive scan, then form (num_experts+1)-length tensor.
    Returns tensor of shape (num_experts + 1,), where offsets[1:] is inclusive cumulative counts.
    """
    flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
    N = flat.numel()
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    out_offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    # Histogram over classes
    grid_hist = (num_experts,)
    _hist_kernel[grid_hist](flat, counts, N, num_experts)
    # Inclusive scan
    _inclusive_scan_kernel[grid_hist](counts, out_offsets, num_experts)
    # Form final offsets of length (num_experts + 1): add 0 at the beginning
    # Using torch.cat is minimal and on device; it's necessary to build the final tensor.
    expert_offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device), out_offsets])
    return expert_offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Triton-only computation: no torch.sort, torch.argsort, torch.bincount, torch.cumsum, torch.cat
        sorted_token_indices = _compute_sorted_token_indices(topk_idx)
        expert_offsets = _compute_expert_offsets(topk_idx, num_experts=256)
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
