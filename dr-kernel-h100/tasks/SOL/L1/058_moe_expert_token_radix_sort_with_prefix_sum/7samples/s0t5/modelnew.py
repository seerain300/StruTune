import torch
import triton
import triton.language as tl


# Triton kernel: count occurrences of each value in x (int32) into counts[0..255]
# x_ptr points to int32 values; counts_ptr points to int32 counts.
@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load values; ensure masked-out lanes don't contribute
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Increment counts for each valid lane. We assume vals in [0, 255].
    # Note: tl.atomic_add on int32 is supported for increments.
    for i in range(0, 256):
        inc = (vals == i) & mask
        tl.atomic_add(counts_ptr + i, inc.to(tl.int32))


# Triton kernel: inclusive prefix sum of a 1D array x into y
# y[i] = sum_{j=0..i} x[j], for i in [0..L-1]
# x_ptr is int32, y_ptr is int64 (we cast on store). We implement a simple loop.
@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.constexpr):
    # Single-program loop to compute inclusive prefix sum
    acc = 0  # int32 accumulator
    for i in range(0, L):
        val = tl.load(x_ptr + i)  # int32
        acc += val
        tl.store(y_ptr + i, acc.to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten topk_idx
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Ensure flat is int32 (get_inputs produces int32)
        # Compute counts with Triton
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # Compute expert offsets via inclusive prefix sum (int64), length 257
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        # Run the prefix sum kernel; L is known at compile time (257). We pass as constexpr.
        inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, L=257)

        # sorted_token_indices: stable argsort of flattened indices (PyTorch)
        # This returns the permutation of [0, N-1] sorted by the values at those indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets