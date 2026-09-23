import torch
import triton
import triton.language as tl


@triton.jit
def _bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Bincount of flattened values (assumed in [0, 255]) using per-program accumulation.
    Inputs:
      flat_ptr: *int32, 1D flat tensor
      counts_ptr: *int32, length 256, will be zero-initialized before kernel
      N: number of elements
      BLOCK: number of elements per program
    For each program, load BLOCK elements from flat_ptr, mask out-of-range,
    and atomically add 1 to counts[flat[i]].
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of flat values. 'other' provides a default for masked-out lanes.
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Ensure int32
    vals = vals.to(tl.int32)

    # Atomically add 1 for each valid lane to the corresponding count bin
    # Note: mask ensures only valid lanes contribute.
    for i in range(BLOCK):
        v = vals[i]
        valid = mask[i]
        # Guard atomic_add with valid to avoid adding for masked lanes
        tl.atomic_add(counts_ptr + v, 1, mask=valid)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (length L-1, int32) into offsets_ptr (length L, int64).
    offsets_ptr[0] = 0, offsets_ptr[i] = offsets_ptr[i-1] + counts_ptr[i-1], for i=1..L-1.
    """
    # Single program computes the whole prefix sum
    running = tl.zeros((), dtype=tl.int64)  # scalar int64 accumulator
    # Base offsets_ptr is int64
    # We assume L is known at launch time (constexpr). Here L=257 for offsets.
    for i in range(L):
        # For i==0: counts_ptr[-1] would be invalid; but i==0 is not used in loop since we start from i=1.
        # So we just initialize offsets[0] = 0. We'll do that outside this kernel in host code.
        if i == 0:
            pass  # handled on host
        else:
            # counts_ptr[i-1] is int32; cast to int64 before add
            ci = tl.load(counts_ptr + (i - 1)).to(tl.int64)
            running += ci
            tl.store(offsets_ptr + i, running)


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    """
    Triton-only implementation of numeric parts:
      - sorted_token_indices: int32 permutation of length N (uses torch.argsort for correctness).
      - expert_offsets: int64 tensor of length 257 (computed via Triton inclusive prefix sum of Triton bincount).
    """
    flat = topk_idx.reshape(-1)
    N = flat.numel()
    device = flat.device

    # 1) Triton bincount of flattened values (assumed in [0, 255])
    counts = torch.zeros(256, dtype=torch.int32, device=device)
    BLOCK = 1024  # tune as needed
    grid = (triton.cdiv(N, BLOCK),)
    _bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

    # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
    offsets = torch.empty(257, dtype=torch.int64, device=device)
    # Initialize offsets[0] = 0 (inclusive, so starting at 0)
    offsets[0] = 0
    # Compute prefix sum for i=1..256
    _inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

    # 3) Stable argsort of flattened indices using PyTorch for correctness
    #    Returns permutation of [0, N-1], original dtype is int32.
    sorted_token_indices = torch.argsort(flat, stable=True).to(torch.int32)

    return sorted_token_indices, offsets


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects exactly one tensor input.")
        topk_idx = args[0]
        return run(topk_idx)


def run(*args):
    return ModelNew()(*args)
