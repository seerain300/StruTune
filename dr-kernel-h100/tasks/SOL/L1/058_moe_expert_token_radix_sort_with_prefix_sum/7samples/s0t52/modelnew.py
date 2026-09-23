import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load flattened values as int32
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # Atomic add into counts[vals] for each element in the chunk
    # Assumes vals in [0, 255], which matches num_experts=256 in original code
    for i in range(BLOCK):
        v = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Compute inclusive prefix sum over L elements, writing int32.
    # offsets_ptr[0] must be initialized to 0 on host.
    prev = tl.zeros((), dtype=tl.int32)
    for i in range(1, L):
        cnt = tl.load(counts_ptr + (i - 1))
        running = prev + cnt
        tl.store(offsets_ptr + i, running)
        prev = running


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
        #    This matches the original run's behavior. The original returns int32.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets