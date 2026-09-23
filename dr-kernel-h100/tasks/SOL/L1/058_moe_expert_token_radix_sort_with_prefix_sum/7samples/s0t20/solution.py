import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Count occurrences of each value in x_ptr (int32) into counts_ptr (int32) of length 256.
    Uses atomic_add per element. Assumes x_ptr values are in [0, 255].
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load flattened indices (int32)
    vals = tl.load(x_ptr + offs, mask=mask, other=0)  # other=0 won't be used due to mask; ensure int32
    # Note: vals are in [0, 255] per get_inputs; no need for bounds checks here.

    # For each bin i in 0..255, count how many vals == i and atomic_add to counts[i].
    # We implement this with a loop over i and masked atomic_add for each lane.
    for i in range(0, 256):
        eq = vals == i
        # Only valid lanes contribute; i is always in range.
        # Atomic add 1 for each True at masked positions.
        tl.atomic_add(counts_ptr + i, tl.where(mask & eq, 1, 0))


@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.int32):
    """
    Compute inclusive prefix sum of x_ptr (int32) into y_ptr (int64), length L.
    Single-program loop over L. Assumes L is small (e.g., 257).
    """
    # y_ptr must be int64
    total = tl.zeros((), dtype=tl.int64)
    # Loop over length L
    for k in range(0, L):
        xk = tl.load(x_ptr + k)  # int32
        total += xk
        tl.store(y_ptr + k, total.to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Compute:
          - sorted_token_indices: stable argsort of flattened indices (PyTorch), int64
          - expert_offsets: inclusive cumsum of counts per expert id, int64, length 257
        """
        # Ensure contiguous and flatten
        flat = topk_idx.reshape(-1).contiguous()

        # sorted_token_indices: stable argsort of values at those flattened positions.
        # Must match original dtype (int64).
        sorted_token_indices = flat.argsort(stable=True).to(torch.int64)

        # Triton bincount: counts per expert id in [0, 255]
        N = flat.numel()
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)

        # Launch Triton kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK)

        # Triton inclusive prefix sum of counts -> expert_offsets (int64), length 257
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
