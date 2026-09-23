import torch
import triton
import triton.language as tl


# Triton kernel: build histogram of int32 values in [0, 255] using atomic adds.
# flat_ptr: *int32, counts_ptr: *int32 (size 256), N: number of elements
@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load values; masked out-of-range lanes get 0 (will not add)
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Convert to int32 in case input is not guaranteed; here flat is int32 already
    vals = vals.to(tl.int32)
    # Atomic add for each valid lane
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton kernel: inclusive prefix sum over a 256-element counts array into offsets[0..256]
# counts_ptr: *int32 (size 256), offsets_ptr: *int32 (size 257), M: 256
@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    # Single program instance; do a simple loop scan
    acc = tl.zeros((), dtype=tl.int32)  # scalar accumulator
    # offsets[0] = 0, offsets[1..256] = inclusive sum up to that index
    # Loop over i from 0 to M-1
    for i in range(0, M):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # 1) Flatten and compute permutation via torch (robust and stable)
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        # sorted_token_indices: permutation of positions that would sort flat ascending, stable=True
        sorted_token_indices = torch.argsort(flat, stable=True)

        # 2) Histogram via Triton (counts per expert id in [0..255])
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](flat, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over counts
        offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets[0] = 0  # initialize first offset
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices.to(torch.int32), offsets