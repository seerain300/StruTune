import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements from the flattened tensor
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load the flattened indices (int32). For out-of-range, ignore via mask.
    idx = tl.load(flat_ptr + offs, mask=mask, other=0)

    # For each bin 0..255, atomically add 1 for each valid idx equal to bin.
    # We use a static loop to avoid dynamic indexing issues in Triton.
    for b in range(256):
        # Only increment for valid lanes and idx == b
        m = mask & (idx == b)
        # Atomic add 1 to counts[b]; Triton supports atomic_add on int32
        tl.atomic_add(counts_ptr + b, tl.sum(m.to(tl.int32)))


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Compute inclusive prefix sum of counts (length L) into offsets.
    # We implement a simple iterative loop with compile-time constant L.
    # offsets_ptr is int64; counts_ptr is int32.
    running = 0
    for i in range(L):
        v = tl.load(counts_ptr + i)
        running += v
        tl.store(offsets_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure topk_idx is contiguous and flattened
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Triton bincount of flattened indices into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # tune as needed; 1024 is a safe default
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        #    This matches the original run's behavior for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
