import torch

# Triton kernels: we need to perform heavy work in Triton, not in PyTorch on device.

# Kernel 1: Histogram of class counts from flat values (int32 expected)
# We assume values are in [0, NUM_CLASSES-1] which here is 0..255.
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # Single program, loop over all elements and atomically add counts.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)  # flat_ptr points to int32
        # Only count if in valid range
        if (val >= 0) & (val < NUM_CLASSES):
            tl.atomic_add(counts_ptr + val, 1)


# Kernel 2: Inclusive scan (prefix sum) over counts to produce start positions
# Out_ptr[0..NUM_CLASSES] inclusive prefix sums. We write start positions for each class.
@triton.jit
def _inclusive_scan_kernel(counts_ptr, out_ptr, NUM_CLASSES: tl.constexpr):
    # Single program sequential inclusive scan
    # out_ptr[0] = 0, out_ptr[k] = out_ptr[k-1] + counts[k-1] for k=1..NUM_CLASSES
    acc = 0
    out_ptr[0] = 0
    for k in range(1, NUM_CLASSES + 1):
        prev = acc
        acc = acc + tl.load(counts_ptr + (k - 1))
        out_ptr[k] = acc
    # We wrote out_ptr[0..NUM_CLASSES]; no return.


# Kernel 3: Global stable argsort based on class; produces indices out_idx (int64)
# We scan classes and for each original index i, place it at the next start position for its class.
@triton.jit
def _global_argsort_stable_kernel(flat_ptr, out_idx_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # out_idx_ptr is int64; scan_ptr is int32 prefix sums (length NUM_CLASSES + 1).
    for c in range(0, NUM_CLASSES):
        start = tl.load(scan_ptr + (c + 0))  # inclusive start for class c
        # Iterate over all original indices i; place those equal to c at out_idx[i] = start + rank
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            if val == c:
                tl.store(out_idx_ptr + i, start)
                start += 1
    # No return value; out_idx_ptr holds sorted indices.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is fixed in this task as 256. We keep it as a constant.
        self.NUM_CLASSES = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is 3D and make it contiguous
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        flat = topk_idx.reshape(-1).contiguous()

        # Ensure flat dtype is int32 for Triton kernel input
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)

        # 1) Compute expert offsets via histogram + inclusive scan (int32 output, length 257)
        counts = torch.zeros(self.NUM_CLASSES, dtype=torch.int32, device=flat.device)
        # Launch histogram kernel (single-program loop)
        _hist_kernel[(1,)](flat, counts, flat.numel(), self.NUM_CLASSES)

        # Compute inclusive scan of counts to get start positions
        scan = torch.empty(self.NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
        _inclusive_scan_kernel[(1,)](counts, scan, self.NUM_CLASSES)

        # expert_offsets: shape (num_experts + 1,) = (257,), int32, with offsets[1:] = scan[1:]
        expert_offsets = torch.empty(self.NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0  # original code sets offsets[0] = 0
        expert_offsets[1:] = scan[1:]  # inclusive prefix sums per class

        # 2) Compute sorted_token_indices via Triton global stable argsort (int64 output, length N)
        N = flat.numel()
        out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
        _global_argsort_stable_kernel[(1,)](flat, out_idx, scan, N, self.NUM_CLASSES)

        # Return with exact shapes/dtypes as required:
        # sorted_token_indices: (N,), dtype int64
        # expert_offsets: (num_experts + 1,) = (257,), dtype int32
        return out_idx, expert_offsets