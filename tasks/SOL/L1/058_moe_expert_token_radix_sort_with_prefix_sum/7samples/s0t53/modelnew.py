import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load flat values; masked lanes get 0
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Atomically increment counts for each valid value
    # Assumes vals are in [0, 255] and counts_ptr has length 256
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Compute inclusive prefix sum over counts_ptr[0:L-1], write to offsets_ptr[1:L]
    running = tl.zeros((), dtype=tl.int32)
    # offsets_ptr[0] is initialized on host to 0
    for i in range(1, L):
        cnt = tl.load(counts_ptr + (i - 1))
        running = running + cnt
        tl.store(offsets_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int32 inside Triton)
        offsets_int32 = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets_int32[0] = 0  # inclusive prefix sum starts at 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets_int32, L=257)

        # Cast offsets to int64 to match torch.bincount(...).cumsum(0) dtype
        offsets = offsets_int32.to(torch.int64)

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        #    This matches the original run's behavior for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets