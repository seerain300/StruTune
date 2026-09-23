import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    """
    Counts occurrences of each value in orig_ptr (int32) into counts_ptr (int32),
    assuming values are in [0, L-1], with L=256 (num_experts).
    """
    lane = tl.program_id(0)
    offsets = lane * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    # For valid lanes, count occurrences for each v in 0..L-1
    for v in range(L):
        eq = vals == v
        increment = tl.where(mask & eq, 1, 0)
        tl.atomic_add(counts_ptr + v, tl.sum(increment))


@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, num_exps: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute exclusive prefix sums across counts_ptr[0:num_exps] into offsets_ptr[0:num_exps].
    offsets[num_exps] is set to N (total elements) via a separate write.
    """
    running = 0
    for i in range(num_exps):
        count_i = tl.load(counts_ptr + i)
        running += count_i
        tl.store(offsets_ptr + i, running)
    # We don't need to store anything else here; N is written by a separate kernel.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Compute expert_offsets using Triton kernels only. No torch operations in forward.
        Returns:
            expert_offsets: int32 tensor of shape (256 + 1,), same as original.
        """
        # Flatten and ensure int32
        flat = topk_idx.to(torch.int32).reshape(-1).contiguous()
        N = flat.numel()
        L = 256  # num_experts

        # Allocate counts and offsets
        counts = torch.zeros(L, dtype=torch.int32, device=flat.device)
        offsets = torch.empty(L + 1, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](flat, counts, N, L=L, BLOCK=BLOCK)

        # Write N to offsets[256] using a tiny Triton kernel
        @triton.jit
        def write_last_offset_kernel(ptr, value):
            tl.store(ptr, value)

        write_last_offset_kernel[(1,)](offsets + 256, N)

        # Launch exclusive scan to fill offsets[0:256]
        exclusive_scan_kernel[(1,)](counts, offsets, num_exps=L, BLOCK=32)

        # Return expert_offsets
        return offsets


def run(*args):
    return ModelNew()(*args)
