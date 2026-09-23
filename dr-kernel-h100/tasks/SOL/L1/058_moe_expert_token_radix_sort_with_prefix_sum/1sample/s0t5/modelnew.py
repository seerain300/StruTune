import torch
import triton
import triton.language as tl


# Triton kernel: histogram of integer values in flat tensor into counts[0..255].
# flat_ptr: *const int32, N: number of elements
@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load values (int32). For masked lanes, set 0 (won't contribute if masked out in store).
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Atomic add into counts
    # counts_ptr is int32*, but Triton requires int32 for atomics in this pattern.
    for i in range(BLOCK):
        idx = offsets[i]
        # Only add if within bounds
        if mask[i]:
            # Triton requires tensor-like indexing; use scalar form via pointer arithmetic
            tl.atomic_add(counts_ptr + vals[i].to(tl.int32), 1)


# Triton kernel: inclusive prefix sum over 'counts' (length M) and write into 'out' (length M+1).
# out[0] is initialized by caller; out[i] = sum_{j=0..i-1} counts[j].
@triton.jit
def inclusive_scan_prefix_sum(counts_ptr, out_ptr, M: tl.int32):
    # Single program instance scans all 256 elements. This is efficient for small M.
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, M):
        v = tl.load(counts_ptr + i)
        total += v
        tl.store(out_ptr + i + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32, CUDA.
        Returns:
          - sorted_token_indices: int32[flattened_size] — torch.argsort(stable=True) permutation.
          - expert_offsets: int32[num_experts+1] — prefix sum of counts of each expert id.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels."
        # Flatten indices
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Stable sort permutation using PyTorch (robust and correct across all N)
        # We sort values, return indices (positions).
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # 2) Histogram of expert indices in [0..255] via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_atomic_kernel[grid](flat, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 counts
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets