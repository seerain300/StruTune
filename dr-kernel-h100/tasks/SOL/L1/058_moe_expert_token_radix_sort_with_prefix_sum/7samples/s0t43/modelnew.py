import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Compute per-expert counts for values in flat_ptr[0:N] assuming values are in [0, 255].
    counts_ptr is int32 of length 256. We do per-program accumulation via atomic_add.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values; other=0 for masked lanes
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)

    # Ensure vals are int32
    vals = vals.to(tl.int32)

    # Accumulate per program into counts[vals] using atomic_add
    # Each program processes BLOCK elements
    for i in range(BLOCK):
        vi = vals[i]
        # valid only if lane is within N
        valid = mask[i]
        # Add 1 for valid lanes
        add_val = 1 if valid else 0
        # Atomic add to the corresponding bin
        tl.atomic_add(counts_ptr + vi, add_val)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int32, length L=256) into offsets_ptr (int64, length L+1).
    offsets[0] = 0, offsets[i] = offsets[i-1] + counts[i-1] for i in [1..L].
    """
    acc = tl.zeros((), dtype=tl.int64)
    # offsets_ptr is int64
    offsets_ptr = offsets_ptr  # Triton doesn't require re-binding here
    offsets_ptr[0] = acc
    for i in range(1, L + 1):
        ci = tl.load(counts_ptr + (i - 1))
        ci = ci.to(tl.int64)
        acc += ci
        offsets_ptr[i] = acc


@triton.jit
def stable_argsort_kernel(flat_ptr, out_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Stable argsort of flattened indices [0..N-1] based on values in flat_ptr.
    Values are in [0, 255]. We implement block-level bitonic sort network.
    out_ptr stores the permutation (sorted indices).
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)  # original indices
    mask = idx < N

    # Load values for these indices
    vi = tl.load(flat_ptr + idx, mask=mask, other=0).to(tl.int32)

    # Initialize out with idx (no mask: out-of-range lanes may be arbitrary, but we won't use them)
    out = idx

    # Bitonic sort network over BLOCK lanes
    # k: size of subsequences being merged (2, 4, 8, ..., BLOCK)
    k = 2
    while k <= BLOCK:
        # j: distance between elements to compare within the current k-block (k//2, k//4, ..., 1)
        j = k // 2
        while j > 0:
            partner = out ^ j  # bitwise XOR gives the 'neighbor' in this stage
            # Only process each pair once: when lane < partner
            do_pair = partner > idx

            # Gather 'other' values for comparison
            v_partner = tl.load(flat_ptr + out ^ partner, mask=do_pair, other=0).to(tl.int32)

            # Ascending if (idx & k) == 0 else descending
            ascending = (idx & k) == 0

            # Compare with stable tie-break by original index: if vi == v_partner, use indices to decide
            less_vi = vi < v_partner
            eq_vi = vi == v_partner
            tie_by_idx = (out < (out ^ partner))  # if equal, smaller original index first for ascending; partner first for descending

            # Determine swap
            swap_asc = less_vi | (eq_vi & tie_by_idx)
            swap_desc = (eq_vi & (~tie_by_idx)) | (vi > v_partner)
            swap = tl.where(ascending, swap_asc, swap_desc)

            # Update out only for do_pair lanes
            # If swap, out = partner; else out = self
            out_update = tl.where(swap, out ^ partner, out)
            # Apply only once per pair
            out = tl.where(do_pair & swap, out ^ partner, out)

            j //= 2
        k *= 2

    # Write permutation
    tl.store(out_ptr + start + tl.arange(0, BLOCK), out, mask=mask)


def _run_triton_only(topk_idx: torch.Tensor):
    """
    Compute:
      - sorted_token_indices: permutation of [0, N-1], int32
      - expert_offsets: int64 tensor of length 257, inclusive prefix sum of bincount over [0,255]
    All computation done in Triton.
    """
    assert topk_idx.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
    # Flatten and ensure contiguous
    flat = topk_idx.reshape(-1).contiguous()
    N = flat.numel()
    device = flat.device

    # 1) Triton bincount: counts per expert id (0..255)
    counts = torch.zeros(256, dtype=torch.int32, device=device)
    BLOCK = 1024  # tuneable
    grid = (triton.cdiv(N, BLOCK),)
    bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

    # 2) Triton inclusive prefix sum for offsets (length 257, int64)
    offsets = torch.empty(257, dtype=torch.int64, device=device)
    inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

    # 3) Triton stable argsort of flattened indices
    sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
    BLOCK_SORT = 1024  # power of two; tuneable
    grid_sort = (triton.cdiv(N, BLOCK_SORT),)
    stable_argsort_kernel[grid_sort](flat, sorted_token_indices, N, BLOCK=BLOCK_SORT)

    return sorted_token_indices, offsets


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    # Fallback: ensure CUDA tensors
    if not topk_idx.is_cuda:
        topk_idx = topk_idx.cuda()
    return _run_triton_only(topk_idx)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects exactly one tensor input.")
        topk_idx = args[0]
        return run(topk_idx)