import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Bincount of int32 values in flat_ptr into counts_ptr[0..255] (int32).
    Each program processes BLOCK elements and performs masked atomic_add.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load flat values; for masked-out lanes, use 0 (neutral for counts)
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # Atomically add 1 for each valid lane to counts[vals]
    # Note: counts_ptr is int32, atomic_add supports int32.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (assumed int32) into offsets_ptr (int64) of length L.
    We cast to int64 inside the kernel to ensure offsets are int64.
    """
    # We'll do a simple sequential loop for L=257 which is fine and robust.
    # Initialize the first offset to 0 (already done by host).
    # For i in 1..L-1:
    #   offsets[i] = offsets[i-1] + counts[i-1]
    # with counts indices shifted by +1.
    # Triton requires loops to be static; we use a Python-side loop (compile-time constant).
    for i in range(1, L):
        prev = tl.load(offsets_ptr + (i - 1))
        val = tl.load(counts_ptr + (i - 1))
        # Cast counts to int64 before addition
        val64 = val.to(tl.int64)
        curr = prev + val64
        tl.store(offsets_ptr + i, curr)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Triton: bincount of expert ids in [0, 255] (int32) and inclusive prefix sum (int64).
        - PyTorch: stable argsort for flattened indices to produce sorted_token_indices.
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount: counts[0..255] as int32
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0  # inclusive, starting at 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
