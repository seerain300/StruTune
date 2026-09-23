import torch
import triton
import triton.language as tl


# Kernel 1: Global stable argsort by class (per-element). Grid over N.
# Each program handles one index i, finds c = flat[i], increments start[c], writes i at position.
@triton.jit
def _global_argsort_fill_kernel(flat_ptr, out_idx_ptr, start_ptr, N, NUM_CLASSES: tl.constexpr):
    i = tl.program_id(axis=0)
    if i >= N:
        return
    val = tl.load(flat_ptr + i)
    c = val  # already int32 in [0, NUM_CLASSES)
    pos = tl.load(start_ptr + c)
    tl.store(out_idx_ptr + pos, i)
    tl.atomic_add(start_ptr + c, 1)


# Kernel 2: Compute per-class counts via histogram. Grid over NUM_CLASSES, each program sets counts[c] = number of occurrences of c.
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    c = tl.program_id(axis=0)
    if c >= NUM_CLASSES:
        return
    cnt = 0
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        if val == c:
            cnt += 1
    tl.store(counts_ptr + c, cnt)


# Kernel 3: Compute inclusive prefix sums of counts (scan) for range [0, NUM_CLASSES). Grid over NUM_CLASSES, single-kernel sequential accumulation.
@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    # Only one program should run. We launch grid=(NUM_CLASSES,), but only program id 0 does the work.
    if tl.program_id(axis=0) != 0:
        return
    running = 0
    for j in range(0, NUM_CLASSES):
        cnt = tl.load(counts_ptr + j)
        running += cnt
        tl.store(scan_ptr + j, running)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed num_experts in this task is 256
        self.NUM_CLASSES = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is 3D and contiguous
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        flat = topk_idx.reshape(-1).contiguous()

        # 1) Build expert offsets via histogram + inclusive scan (int32, length 257)
        counts = torch.zeros(self.NUM_CLASSES, dtype=torch.int32, device=flat.device)
        # Run histogram kernel: grid over NUM_CLASSES
        _hist_kernel[(self.NUM_CLASSES,)](flat, counts, flat.numel(), self.NUM_CLASSES)
        # Compute inclusive scan on counts to get scan[0..255]
        scan = torch.zeros(self.NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
        _inclusive_scan_kernel[(self.NUM_CLASSES,)](counts, scan, self.NUM_CLASSES)
        # expert_offsets: (num_experts + 1,) = (257,), int32. original sets offsets[0] = 0 and [1:] via cumsum
        expert_offsets = torch.empty(self.NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = scan[1:]  # inclusive prefix sums per class

        # 2) Build sorted_token_indices via Triton global stable argsort (int64, length N)
        N = flat.numel()
        out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
        # Launch per-element kernel over grid (N,)
        _global_argsort_fill_kernel[(N,)](flat, out_idx, scan, N, self.NUM_CLASSES)

        # Return with exact shapes/dtypes:
        # sorted_token_indices: (N,), dtype int64
        # expert_offsets: (num_experts + 1,) = (257,), dtype int32
        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
