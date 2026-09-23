import torch
import triton
import triton.language as tl


@triton.jit
def triton_bincount_kernel(x_ptr, counts_ptr, N, NUM_BINS: tl.constexpr, BLOCK: tl.constexpr):
    """
    Triton kernel to compute per-bin counts for int32 values in x_ptr[0:N].
    Assumes valid values in [0, NUM_BINS-1]. We guard with a mask and only perform atomic_add for valid ids.
    counts_ptr is int32, length NUM_BINS.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load x as int32
    x = tl.load(x_ptr + offs, mask=mask, other=0)

    # Valid ids are in [0, NUM_BINS-1]
    valid = mask & (x >= 0) & (x < NUM_BINS)

    # For valid ids, atomic add 1 into counts[x]
    # Note: Triton supports atomic_add on int32.
    # We convert x to int32 for indexing.
    # Only do atomic_add for valid positions.
    x_i32 = x.to(tl.int32)
    # We need to perform atomic add per valid element. Triton allows elementwise operations with masks.
    # Atomic add expects per-lane pointer. We create per-lane offsets: counts_ptr + x_i32.
    # Only valid lanes will perform the add; invalid lanes do nothing.
    # To avoid illegal memory access, we set x to 0 for invalid lanes so counts_ptr + x is safe.
    x_valid = tl.where(valid, x_i32, 0)
    # Now atomic add 1 for each valid element.
    # Use mask to avoid adding for invalid lanes.
    tl.atomic_add(counts_ptr + x_valid, 1, mask=valid)


@triton.jit
def triton_inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.constexpr):
    """
    Triton kernel to compute inclusive prefix sum of x_ptr[0:L] and write to y_ptr[0:L].
    x_ptr is int32, y_ptr is int64.
    We perform a simple loop in a single program:
      for i in range(L): y[i] = sum_{j=0..i} x[j]
    """
    # Single-program grid
    # Use a while-like loop with static bound using constexpr L
    acc = 0  # int32 accumulator
    i = 0
    while i < L:
        val = tl.load(x_ptr + i)  # int32
        acc += val
        # Write inclusive sum as int64
        tl.store(y_ptr + i, acc.to(tl.int64))
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of run:
        - sorted_token_indices: int32 permutation of [0, N-1] with stable argsort (PyTorch)
        - expert_offsets: int64 of shape (257,) computed as Triton bincount + Triton inclusive prefix sum
        """
        # Ensure contiguous and int32
        if not topk_idx.is_cuda:
            # Triton requires CUDA; move to GPU if necessary
            topk_idx = topk_idx.to('cuda')

        topk_idx = topk_idx.contiguous().to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # sorted_token_indices: PyTorch stable argsort (int32)
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # Triton bincount for per-expert counts
        NUM_BINS = 256
        counts = torch.zeros(NUM_BINS, dtype=torch.int32, device=flat.device)

        # Launch Triton bincount kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        triton_bincount_kernel[grid](flat, counts, N, NUM_BINS, BLOCK)

        # Triton inclusive prefix sum to produce expert_offsets (int64, length 257)
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        # We only write first 257 elements; offset[0] should be 0. y_ptr[0] will be left undefined if we don't set it.
        # Compute prefix sum into a temporary int64 tensor of length 256, then copy into expert_offsets[1:].
        prefix = torch.empty(NUM_BINS, dtype=torch.int64, device=flat.device)
        triton_inclusive_prefix_sum_kernel[(1,)](counts, prefix, NUM_BINS)

        # Write into final expert_offsets: offset[0] = 0, offset[1:] = prefix
        expert_offsets[0] = 0
        if NUM_BINS > 0:
            expert_offsets[1:] = prefix

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
