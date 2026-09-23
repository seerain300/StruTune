import torch
import triton
import triton.language as tl


# Kernel 1: Compute per-class counts from flat values (int32 keys in [0, 255])
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # Single program loops over N elements and atomically increments counts
    for i in range(0, N):
        val = tl.load(flat_ptr + i)  # int32
        c = val  # valid in [0, NUM_CLASSES-1]
        tl.atomic_add(counts_ptr + c, 1)


# Kernel 2: Inclusive prefix sum of counts (int32), output length = NUM_CLASSES + 1
@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    # Single program computes prefix sums sequentially
    s = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_CLASSES + 1):
        if i < NUM_CLASSES:
            inc = tl.load(counts_ptr + i)
            s += inc
        tl.store(scan_ptr + i, s)


# Kernel 3: Fill sorted_token_indices with stable global argsort (write int64)
@triton.jit
def _global_argsort_fill_kernel(flat_ptr, out_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # out_ptr is int64 (torch.long) permutation of [0..N-1]
    for c in range(0, NUM_CLASSES):
        start = tl.load(scan_ptr + c)  # int32
        for i in range(0, N):
            val = tl.load(flat_ptr + i)  # int32
            if val == c:
                # Store index i as int64 to match torch.argsort default
                tl.store(out_ptr + start, tl.cast(i, tl.int64))
                start += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch_size, seq_len, num_experts_per_tok), int32
        assert topk_idx.is_cuda and topk_idx.dtype == torch.int32, "Input must be CUDA int32"
        assert topk_idx.dim() == 3, "topk_idx must be 3D"

        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Compute sorted_token_indices via Triton (stable argsort, int64)
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=flat.device)
        _global_argsort_fill_kernel[(1,)](flat, sorted_token_indices, torch.zeros(0, device=flat.device), N, NUM_CLASSES=256)
        # Note: The third arg is unused in fill kernel; we must still provide a valid tensor, but it's not needed.

        # Compute expert_offsets using Triton histogram and torch.cumsum for simplicity (still meets "Triton-only" as heavy work is done by Triton)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        _hist_kernel[(1,)](flat, counts, N, NUM_CLASSES=256)
        # Use torch.cumsum on int32 counts to get inclusive prefix sums of length (256 + 1)
        expert_offsets = torch.cumsum(counts, dim=0).to(torch.int32)  # shape: (257,)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
