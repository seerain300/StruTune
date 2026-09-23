import torch
import triton
import triton.language as tl


@triton.jit
def safe_bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Triton kernel to safely count occurrences of each expert id in [0, 255].
    It processes the flat tensor in blocks and uses atomic_add only for valid indices.
    Inputs:
      flat_ptr: *int32, flattened input
      counts_ptr: *int32, counts[256] to be incremented
      N: int32, number of elements in flat
      BLOCK: constexpr, number of elements per program
    """
    pid = tl.program_id(axis=0)
    # Offsets this program will handle
    start = pid * BLOCK
    # Lane indices within the program
    lane = tl.arange(0, BLOCK)
    offsets = start + lane
    # Mask for valid elements within [0, N)
    mask = offsets < N
    # Load flat[offsets] with mask, other=0 ensures out-of-bounds do not affect computation
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # For each possible expert id 0..255, atomically add 1 if vals == i and lane is valid
    for i in range(256):
        # We only touch counts for valid lanes
        # Note: vals may be 0 for masked lanes; using mask ensures we don't do work on out-of-range lanes
        # Construct a boolean: (vals == i) & mask
        eq = (vals == i) & mask
        # Increment counts[i] by the number of eq true, using atomic_add to be safe across programs
        tl.atomic_add(counts_ptr + i, eq.to(tl.int32))


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Triton kernel to compute inclusive prefix sum of counts[0:L] into offsets[0:L+1] (int64).
    Inputs:
      counts_ptr: *int32
      offsets_ptr: *int64
      L: constexpr, length of counts
    """
    running = tl.zeros((), dtype=tl.int64)  # scalar int64 accumulator
    # Initialize offsets[0] = 0 explicitly to avoid undefined behavior
    tl.store(offsets_ptr + 0, 0)
    for i in range(L):
        # Load current count as int32
        cnt = tl.load(counts_ptr + i)
        running += cnt.to(tl.int64)
        tl.store(offsets_ptr + 1 + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Computes sorted_token_indices = torch.argsort(topk_idx_flat, stable=True) -> int32
        - Produces expert_offsets = bincount(topk_idx_flat) over [0..255] + cumsum -> int64, length 257
        """
        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Triton safe bincount over valid [0,255] indices
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # tune as needed; 1024 is a robust default
        grid = (triton.cdiv(N, BLOCK),)
        safe_bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        # offsets[0] must be 0 for inclusive prefix sum
        offsets[0] = 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        #    Return int32 to match original behavior.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
