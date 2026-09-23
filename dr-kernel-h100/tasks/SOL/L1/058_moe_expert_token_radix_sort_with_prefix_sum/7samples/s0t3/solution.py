import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute per-expert counts (length 256) from flattened int32 indices.
    For each element in x_ptr[0:n_elements], if value v in [0, 255], atomically add 1 to counts[v].
    x_ptr: flattened indices (int32)
    counts_ptr: per-expert counts (int32, length 256)
    n_elements: total number of elements in the flattened array
    """
    # Each program handles a block of elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    # Load a block of values from x_ptr
    # Note: x_ptr is int32; load as int32
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)  # int32

    # For each possible expert id 0..255, atomically add 1 for valid positions where vals == i
    for i in range(256):
        # Create a mask for positions where vals == i
        m = (vals == i) & mask
        # Accumulate count of True's in this block; since m is boolean, cast to int32 and sum.
        # Triton provides tl.sum over a vector. This is safe because BLOCK is constexpr.
        cnt = tl.sum(m.to(tl.int32), axis=0)
        # Atomic add to counts[i]
        tl.atomic_add(counts_ptr + i, cnt)


@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, length: tl.constexpr):
    """
    Compute inclusive prefix sum of x_ptr (int32) into y_ptr (int64), length is known at compile time.
    Single program does this sequentially. length is assumed to be 257 (num_experts + 1).
    """
    # Initialize y[0] = x[0]
    y0 = tl.load(x_ptr + 0)
    tl.store(y_ptr + 0, y0.to(tl.int64))

    # Compute and store inclusive sums for i in [1, length-1]
    for i in range(1, length):
        xi = tl.load(x_ptr + i)
        yi_prev = tl.load(y_ptr + (i - 1))
        yi = yi_prev + xi
        tl.store(y_ptr + i, yi.to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Accepts topk_idx: (batch_size, seq_len, num_experts_per_tok), int32, on device.
        Returns:
          - sorted_token_indices: permutation of [0, N-1] (N = number of elements), int32
          - expert_offsets: inclusive prefix sum of counts for 256 experts, shape (257,), int64
        """
        # Ensure contiguous 1D flattened view
        flat = topk_idx.reshape(-1).contiguous()

        # Number of elements
        n = flat.numel()

        # 1) Compute per-expert counts via Triton bincount. Vector length 256 (num_experts = 256).
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)

        # Choose a block size; 1024 is a good default for atomic-heavy kernels
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)

        # Launch Triton bincount kernel
        bincount_kernel[grid](flat, counts, n_elements=n, BLOCK=BLOCK)

        # 2) Compute inclusive prefix sum of counts to get expert_offsets in int64, length 257
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)

        # Launch single-program prefix sum kernel. Length is constexpr 257.
        inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, length=257)

        # 3) sorted_token_indices: stable argsort of flattened positions (PyTorch), as in original run
        # This returns the permutation of [0, N-1] sorted by the values at those indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
