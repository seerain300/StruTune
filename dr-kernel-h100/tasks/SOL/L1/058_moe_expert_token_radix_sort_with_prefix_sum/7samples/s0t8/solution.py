import torch
import triton
import triton.language as tl


# Triton kernel: per-element atomic bincount into counts[0..255].
# Assumes input values are int32, and valid ids are in [0, 255].
@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load values; invalid lanes get 0 to avoid undefined behavior
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Atomic add 1 for each valid value in [0, 255]
    for i in range(0, BLOCK):
        val = vals[i]
        valid = mask[i] & (val >= 0) & (val <= 255)
        if valid:
            tl.atomic_add(counts_ptr + val, 1)


# Triton kernel: inclusive prefix sum over L elements (int32 counts -> int64 output)
@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.constexpr):
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, L):
        val = tl.load(x_ptr + i)  # int32
        acc += val
        tl.store(y_ptr + i, acc.to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten topk_idx to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Compute sorted_token_indices using PyTorch stable argsort
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # Prepare counts vector in Triton
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # Compute expert offsets (int64, length 257) via Triton inclusive prefix sum
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, L=257)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
