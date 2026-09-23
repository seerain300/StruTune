import torch
import triton
import triton.language as tl


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 256  # constant in the original setup

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is 3D and contiguous
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        flat = topk_idx.reshape(-1).contiguous()

        N = flat.numel()

        # 1) Compute per-class counts via Triton (histogram)
        counts = torch.zeros(self.num_classes, dtype=torch.int32, device=flat.device)
        _hist_kernel[(1,)](flat, counts, N, self.num_classes)

        # 2) Compute inclusive prefix sums of counts via Triton (scan)
        scan = torch.empty(self.num_classes + 1, dtype=torch.int32, device=flat.device)
        _inclusive_scan_kernel[(1,)](counts, scan, self.num_classes)

        # 3) Compute sorted_token_indices via Triton stable fill (int64 to match torch.argsort default)
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=flat.device)
        _global_argsort_fill_kernel[(1,)](flat, sorted_token_indices, scan, N, self.num_classes)

        # 4) expert_offsets: return (num_experts + 1,) int32, same as original
        expert_offsets = scan[1:]  # shape: (256 + 1,) = (257,), int32

        return sorted_token_indices, expert_offsets


# Triton kernels: Triton-only heavy computation
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # Single program iterates over N elements and atomically increments counts
    for i in range(0, N):
        val = tl.load(flat_ptr + i)  # int32
        c = val  # valid in [0, NUM_CLASSES-1]
        tl.atomic_add(counts_ptr + c, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    # Single program computes inclusive prefix sum of counts
    s = 0
    for i in range(0, NUM_CLASSES + 1):
        # When i == NUM_CLASSES, s remains last sum; we don't store it.
        s = tl.load(counts_ptr + i) if i < NUM_CLASSES else s
        # Use scalar operations; Triton will vectorize if we use parallel grid, but we use single program here.
        tl.store(scan_ptr + i, s)


@triton.jit
def _global_argsort_fill_kernel(flat_ptr, out_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # Fill out_ptr with stable argsort indices
    for c in range(0, NUM_CLASSES):
        start = tl.load(scan_ptr + c)
        # Iterate original indices in increasing order to ensure stability
        for i in range(0, N):
            val = tl.load(flat_ptr + i)
            if val == c:
                tl.store(out_ptr + start, tl.cast(i, tl.int64))
                start += 1
    # Remaining scan_ptr entries beyond NUM_CLASSES are not written; we only need scan[0..NUM_CLASSES]


# Example usage when integrated into a larger harness:
# model = ModelNew().cuda()
# topk_idx = torch.randint(0, 256, (8, 256, 8), dtype=torch.int32, device='cuda')
# sorted_token_indices, expert_offsets = model(topk_idx)