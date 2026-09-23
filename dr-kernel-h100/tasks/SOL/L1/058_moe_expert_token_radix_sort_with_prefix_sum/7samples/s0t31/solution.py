import triton
import triton.language as tl


@triton.jit
def _bincount_kernel(flat_ptr: tl.pointer_type(tl.int64), counts_ptr: tl.pointer_type(tl.int32), N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load flattened values as int64 to avoid dtype issues in Triton
    val = tl.load(flat_ptr + offs, mask=mask, other=0).to(tl.int64)
    # Only count if 0 <= val <= 255
    valid = (val >= 0) & (val <= 255) & mask
    idx = val.to(tl.int32)
    # Atomic add 1 for each valid element
    tl.atomic_add(counts_ptr + idx, 1, mask=valid)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr: tl.pointer_type(tl.int32), offsets_ptr: tl.pointer_type(tl.int64), L: tl.constexpr):
    # Compute inclusive prefix sum: offsets[j] = sum_{k=0..j-1} counts[k] for j = 0..L-1
    # offsets[0] = 0 (already initialized)
    total = tl.zeros((), dtype=tl.int64)
    for j in range(1, L):
        c = tl.load(counts_ptr + j)
        total += c.to(tl.int64)
        tl.store(offsets_ptr + j, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # 1) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # Convert flat to int64 for Triton load; avoids dtype mismatch in kernel
        flat64 = flat.to(torch.int64)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _bincount_kernel[grid](flat64, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=device)
        offsets[0] = 0  # inclusive, starting at 0
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
