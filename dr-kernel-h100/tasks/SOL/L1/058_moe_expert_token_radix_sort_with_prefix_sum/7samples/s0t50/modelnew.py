import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program handles a contiguous chunk of BLOCK elements.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load flattened values; mask ensures we don't read out of bounds.
    vals = tl.load(x_ptr + offs, mask=mask, other=0)  # vals are int32
    # Atomically accumulate counts for each element in [0, 255]
    # Note: we assume input vals are in [0, 255], matching original behavior.
    for i in range(256):
        # Build a mask of positions where vals == i
        eq_mask = mask & (vals == i)
        # For those positions, increment counts[i]
        # Atomic add expects int32
        tl.atomic_add(counts_ptr + i, eq_mask.to(tl.int32).sum())


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Single-program kernel computing inclusive prefix sum of the first L elements.
    # offsets_ptr has length L+1, offsets[0]=0, offsets[1..] = cumsum(counts[0..L-1]).
    # We implement a simple iterative accumulation.
    # Note: L must be small (here 257), so a loop is fine and robust.
    running = tl.zeros((), dtype=tl.int64)  # int64 accumulator
    # offsets_ptr is int64
    # We will write offsets[1..L] one by one, and keep offsets[0] = 0
    for i in range(1, L + 1):
        # sum of counts[0..i-1] needs reading previous offsets or recomputing.
        # Since L is small and we can afford recomputation, we compute directly:
        # But recomputing here is not practical in Triton. Instead, we maintain a running sum.
        # Approach: offsets_ptr[i] = previous_offset + counts[i-1]
        prev = running
        # We need counts[i-1]; load scalar
        cnt = tl.load(counts_ptr + (i - 1))
        running = prev + cnt.to(tl.int64)
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

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0  # inclusive prefix sum starts at 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        #    This matches the original run's behavior for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets