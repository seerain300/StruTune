import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic_kernel(values_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    # Each program processes a chunk of elements
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values (int32). Use 0 for out-of-range.
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)

    # Atomic add into counts[vals]
    # counts_ptr is int32*, vals are int32. Bounds are guaranteed since vals in [0, 255].
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    # Single program performs inclusive scan over M=256
    acc = 0
    for i in range(0, M):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)  # offsets[1..] = prefix sums
    # offsets[0] is kept as zero by host before launch


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args should contain topk_idx: shape (B, S, EPT), int32, CUDA
        topk_idx = args[0]
        device = topk_idx.device

        # Flatten to 1D
        flat = topk_idx.reshape(-1)

        # 1) Compute permutation via PyTorch (guarantees correctness identical to original)
        # sorted_token_indices is the permutation that would sort flat ascending, stable.
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        N = flat.numel()
        # Choose a reasonable block size; 1024 works well across typical sizes
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](flat, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        # Initialize offsets[0] to 0; kernel will fill offsets[1..]
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
