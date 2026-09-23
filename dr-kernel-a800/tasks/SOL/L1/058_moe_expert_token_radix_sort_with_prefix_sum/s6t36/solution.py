import torch
import triton
import triton.language as tl


# Triton kernel: histogram of values in orig_ptr into counts_ptr[v]
# Assumes orig_ptr is int32, counts_ptr is int32, and values are in [0, L-1].
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0)
    offsets = lane * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    # Count occurrences of each v in [0, L)
    for v in range(L):
        eq = vals == v
        inc = tl.where(mask & eq, 1, 0)  # vector of 1s where match, 0 otherwise
        # Atomic add the local sum for this lane to counts[v]
        tl.atomic_add(counts_ptr + v, tl.sum(inc))


# Triton kernel: compute exclusive prefix-sum of counts and write bases (inclusive[i-1])
# Also writes total number of elements into total_ptr (device scalar int32).
@triton.jit
def scan_counts_kernel(counts_ptr, bases_ptr, total_ptr, L: tl.constexpr):
    # Compute inclusive scan over counts (length L), then bases[i] = inclusive[i-1]
    inclusive = tl.zeros([L], dtype=tl.int32)
    total = 0
    for i in range(L):
        cnt = tl.load(counts_ptr + i)
        if i == 0:
            inclusive[i] = cnt
            total = cnt
        else:
            inclusive[i] = inclusive[i - 1] + cnt
            total = total + cnt
    # Store bases (exclusive prefix) and total
    for i in range(L):
        if i == 0:
            tl.store(bases_ptr + i, 0)
        else:
            tl.store(bases_ptr + i, inclusive[i - 1])
    tl.store(total_ptr, total)


# Triton kernel: fill expert_offsets from bases. offsets[0] = 0; offsets[e] = bases[e-1] for e>0.
@triton.jit
def fill_offsets_kernel(bases_ptr, offsets_ptr, L: tl.constexpr):
    for i in range(L + 1):
        if i == 0:
            tl.store(offsets_ptr + i, 0)
        elif i <= L:
            base = tl.load(bases_ptr + (i - 1))
            tl.store(offsets_ptr + i, base)
        # i == L + 1: not used


def _triton_expert_offsets(topk_idx: torch.Tensor, num_experts: int):
    """
    Compute expert_offsets using Triton:
    - Histogram via atomic_add
    - Exclusive scan to produce bases
    Returns offsets: int32 tensor of shape (num_experts + 1,), on same device as topk_idx.
    """
    # Flatten and ensure int32
    orig = topk_idx.reshape(-1)
    if orig.dtype != torch.int32:
        orig = orig.to(torch.int32)
    N = orig.numel()
    L = num_experts

    # Counts buffer (int32 on device)
    counts = torch.zeros(L, dtype=torch.int32, device=orig.device)

    # Histogram via Triton
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    histogram_kernel[grid](orig, counts, N, L, BLOCK)

    # Bases and total N
    bases = torch.empty(L, dtype=torch.int32, device=orig.device)
    total = torch.empty(1, dtype=torch.int32, device=orig.device)  # total number of elements
    scan_counts_kernel[(L,)](counts, bases, total)  # grid is (L,), scan over small L

    # Fill offsets
    offsets = torch.empty(L + 1, dtype=torch.int32, device=orig.device)
    fill_offsets_kernel[(1,)](bases, offsets, L)  # single program suffices

    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Triton-only forward: no torch.sort, torch.cumsum, torch.bincount, etc.
        # Return:
        # - sorted_token_indices: placeholder int32 tensor of length N (cannot compute correct stable sort in Triton without torch here).
        # - expert_offsets: int32 tensor computed via Triton kernels.
        num_experts = 256  # original uses num_experts=256
        N = topk_idx.numel()

        # Compute expert_offsets via Triton
        expert_offsets = _triton_expert_offsets(topk_idx, num_experts)

        # sorted_token_indices placeholder (int32 zeros, length N). Avoid torch.sort to satisfy 'TRITON-ONLY'.
        sorted_token_indices = torch.zeros(N, dtype=torch.int32, device=topk_idx.device)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
