import torch
import triton
import triton.language as tl


# Triton kernel: bincount of values in x (int32). Assumes x[i] in [0, 255].
# We launch a 1D grid; each program instance processes BLOCK elements.
@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load a chunk; for masked-out lanes, set value to -1 so atomic_add is a no-op
    vals = tl.load(x_ptr + offsets, mask=mask, other=-1)
    # Iterate over the chunk using a constexpr loop and atomic_add per valid lane
    for i in range(0, BLOCK):
        val = vals[i]
        # Only increment if val in [0, 255] and lane is valid
        is_valid = (val >= 0) & (val <= 255) & mask[i]
        # Atomic add 1 to counts[val]; if val out of range, do nothing (we masked other=-1)
        tl.atomic_add(counts_ptr + val, 1, mask=is_valid)


# Triton kernel: inclusive prefix sum of a 1D array x (int32), writes y (int64) of length L.
@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.constexpr):
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, L):
        val = tl.load(x_ptr + i)  # int32
        acc += val
        tl.store(y_ptr + i, acc.to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten topk_idx to int32 (get_inputs produces int32)
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Compute counts via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # Compute expert offsets (int64, length 257) via Triton inclusive prefix sum
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, L=257)

        # sorted_token_indices: stable argsort of flattened indices
        # This returns the permutation of [0, N-1] sorted by the values at those indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets