import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # count occurrences of each class in flat
    # NUM_CLASSES = 256
    for i in range(0, N):
        val = tl.load(flat_ptr + i)  # int32
        if (val >= 0) & (val <= 255):
            # atomic add to counts[val]
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    # Inclusive prefix sum of counts: scan[j+1] = sum_{k=0..j} counts[k]
    running = 0
    for j in range(0, NUM_CLASSES + 1):
        running += tl.load(counts_ptr + j)
        tl.store(scan_ptr + j, running)


@triton.jit
def _global_argsort_fill_kernel(flat_ptr, out_idx_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    # Fill out_idx with stable global argsort: for each class c, place indices in original order
    # out_idx: int64
    # scan: int32 (exclusive scan at j gives start position for class j)
    for c in range(0, NUM_CLASSES + 1):  # loop guard; only 0..255 are used
        start = tl.load(scan_ptr + c)  # int32
        end = tl.load(scan_ptr + (c + 1))  # int32
        for i in range(0, N):
            val = tl.load(flat_ptr + i)  # int32
            if (val == c):
                # Place i at position start; increment start for next
                # out_idx_ptr is int64
                pos = start + tl.zeros((), dtype=tl.int32)  # int32
                tl.store(out_idx_ptr + pos, tl.cast(i, tl.int64))
                start += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from the original code
        self.num_classes = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is 3D as in original helper
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D; Triton expects int32 for indices
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Compute histogram of expert IDs via Triton
        counts = torch.zeros(self.num_classes, dtype=torch.int32, device=flat.device)
        # Triton kernel: single-program loop over N
        # Note: We'll launch grid=(1,) and iterate all N in the kernel.
        _hist_kernel[(1,)](flat, counts, N, self.num_classes)

        # 2) Compute inclusive prefix sums (scan) of counts to get offsets
        scan = torch.empty(self.num_classes + 1, dtype=torch.int32, device=flat.device)
        _inclusive_scan_kernel[(1,)](counts, scan, self.num_classes)

        # 3) Compute sorted_token_indices via Triton stable counting sort fill
        # Allocate output permutation as int64 to match torch.argsort default
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=flat.device)
        _global_argsort_fill_kernel[(1,)](flat, sorted_token_indices, scan, N, self.num_classes)

        # 4) expert_offsets: original sets length (num_experts + 1) and uses cumulative counts
        #    From our scan[1:], we have inclusive cumulative counts for original flat.
        #    Return scan[1:] with shape (num_experts + 1,) = (257,), int32
        expert_offsets = scan[1:]

        # Return exactly as original: sorted_token_indices (N,), int64, device same as input
        # and expert_offsets (num_experts + 1,), int32, device same as input
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
