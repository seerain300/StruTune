import torch
import triton
import triton.language as tl


@triton.jit
def _global_counting_sort_stable_kernel(
    flat_ptr,          # *int32, input values (flattened)
    out_idx_ptr,       # *int32, output permutation (flattened)
    offsets_ptr,       # *int32, per-class offsets (length 256), side-effect
    N,                 # int32, total number of elements in flat
    NUM_CLASSES: tl.constexpr,  # number of classes, here 256
):
    i = tl.program_id(0)
    if i < N:
        # Load value at position i
        val = tl.load(flat_ptr + i)
        # Compute class index (assumes val in [0, NUM_CLASSES-1])
        # For our use, val is an expert index in [0, 255].
        cls = val
        # Get current position for this class
        pos = tl.load(offsets_ptr + cls)
        # Write i into out_idx at position 'pos'
        tl.store(out_idx_ptr + pos, i)
        # Advance the offset for this class
        tl.store(offsets_ptr + cls, pos + 1)


@triton.jit
def _hist_kernel(
    flat_ptr,           # *int32
    counts_ptr,         # *int32, output counts (length NUM_CLASSES)
    N,                  # int32
    NUM_CLASSES: tl.constexpr,
):
    cls = tl.program_id(0)
    if cls < NUM_CLASSES:
        cnt = tl.zeros((), dtype=tl.int32)
        # Loop over all elements
        for j in range(0, N):
            val = tl.load(flat_ptr + j)
            if val == cls:
                cnt += 1
        tl.store(counts_ptr + cls, cnt)


@triton.jit
def _inclusive_scan_kernel(
    counts_ptr,         # *int32
    out_ptr,            # *int32
    NUM_CLASSES: tl.constexpr,
):
    # Simple sequential inclusive scan per class; grid size should be NUM_CLASSES.
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_CLASSES):
        cnt = tl.load(counts_ptr + i)
        running += cnt
        tl.store(out_ptr + i, running)


def _compute_sorted_token_indices(flat: torch.Tensor) -> torch.Tensor:
    """
    Triton-based stable global counting sort for values in [0, 255].
    Returns the permutation of indices that would sort flat ascending.
    """
    # Ensure int32 for Triton
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    offsets = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Launch one program per element
    grid = (N,)
    _global_counting_sort_stable_kernel[grid](flat, out_idx, offsets, N, 256)
    return out_idx


def _compute_expert_offsets(topk_idx: torch.Tensor, num_experts: int = 256) -> torch.Tensor:
    """
    Compute per-expert offsets using Triton:
    - Histogram of original flat values.
    - Inclusive prefix sum to get offsets[1:].
    - Return offsets of length (num_experts + 1), with offsets[0] = 0.
    """
    flat = topk_idx.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    out_offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    grid = (num_experts,)
    _hist_kernel[grid](flat, counts, flat.numel(), num_experts)
    _inclusive_scan_kernel[grid](counts, out_offsets, num_experts)
    expert_offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device), out_offsets])
    return expert_offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # We must use Triton for all computation; no torch.sort / torch.argsort / torch.bincount / torch.cumsum.
        # Flatten for sorting (same as original run).
        flat = topk_idx.reshape(-1).contiguous()
        # Triton-based global stable sort to match torch.argsort(stable=True) for values in [0, 255].
        sorted_token_indices = _compute_sorted_token_indices(flat)
        # Triton-based expert offsets
        expert_offsets = _compute_expert_offsets(topk_idx, num_experts=256)
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
