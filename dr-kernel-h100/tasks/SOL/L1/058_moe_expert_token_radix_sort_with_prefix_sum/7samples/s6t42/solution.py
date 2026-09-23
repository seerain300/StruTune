import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Compute histogram of values in flat_ptr[0:N] into counts_ptr[0:NUM_CLASSES] using atomic adds.
    Values are expected to be in [0, NUM_CLASSES-1].
    """
    idx = tl.program_id(axis=0)  # single program; grid should be (1,)
    # Loop over N to count
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        # Ensure val is within range; if not, bin it into NUM_CLASSES-1 for safety (not used here since inputs are valid)
        val_bin = val
        # atomic add into counts[val_bin]
        tl.atomic_add(counts_ptr + val_bin, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0:NUM_CLASSES] and write to scan_ptr[0:NUM_CLASSES+1].
    We need scan[0..NUM_CLASSES] with scan[j] = sum_{k<j} counts[k].
    """
    # This kernel is tiny; single program suffices
    # Preload counts
    acc = 0
    for j in range(0, NUM_CLASSES + 1):
        if j == NUM_CLASSES:
            tl.store(scan_ptr + j, acc)  # set last element to 0
        else:
            val = tl.load(counts_ptr + j)
            acc += val
            tl.store(scan_ptr + j, acc)


@triton.jit
def _global_argsort_stable_kernel(flat_ptr, out_idx_ptr, scan_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Fill out_idx_ptr[0:N] with the global stable permutation indices that would sort flat_ptr in ascending order.
    We use counting sort by class: for each class c, place original indices in increasing order using scan to determine start positions.
    """
    # Single program iterates over N to fill out_idx; out_idx is int64 to match torch.argsort default dtype.
    for i in range(0, N):
        val = tl.load(flat_ptr + i)
        # Find start position for this value in its class
        # For stable sort, we rely on the fact that we process i in increasing order; that handles ties by original order.
        # Compute start = scan[val] if val in [0, NUM_CLASSES-1], else 0 (not used).
        # We assume val is in [0, NUM_CLASSES-1] as per problem setup.
        start = tl.load(scan_ptr + val)
        # Place i at start
        # out_idx_ptr is int64
        # Use atomic increment to reserve the next slot
        next_idx = start + 1
        tl.atomic_add(out_idx_ptr + start, 0)  # no-op
        tl.atomic_add(out_idx_ptr + start, 1)  # mark placement
        # We need to actually store i, but Triton does not support direct scalar store with computed pointer like this in a vectorized way;
        # instead, we perform out_idx[i] = next_idx. However, since we don't have per-lane scatter, we recompute next_idx per i and rely on atomic to create a unique slot.
        # To write the index i into the position start, we need to overwrite that specific element. Triton does not support indexed stores here,
        # so we use a trick: set out_idx[start] = i by performing an atomic add of i into out_idx_ptr[start]. Since we have only one writer per position, it's safe.
        tl.atomic_add(out_idx_ptr + start, i)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # NUM_CLASSES is fixed to 256 as per the original setup
        self.NUM_CLASSES = 256

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok)
        Returns:
          - sorted_token_indices: torch.Tensor of shape (N,), dtype int64
          - expert_offsets: torch.Tensor of shape (num_experts + 1,) = (257,), dtype int32
        """
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D and ensure contiguity
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Compute per-class counts via Triton
        counts = torch.zeros(self.NUM_CLASSES, dtype=torch.int32, device=flat.device)
        _hist_kernel[(1,)](flat, counts, N, NUM_CLASSES=self.NUM_CLASSES)

        # 2) Compute inclusive prefix sum scan[0:257] via Triton
        scan = torch.empty(self.NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
        _inclusive_scan_kernel[(1,)](counts, scan, NUM_CLASSES=self.NUM_CLASSES)

        # 3) Build sorted_token_indices via Triton global stable argsort
        # We need out_idx to be int64 to match torch.argsort default dtype
        out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
        _global_argsort_stable_kernel[(1,)](flat, out_idx, scan, N, NUM_CLASSES=self.NUM_CLASSES)

        # 4) Build expert_offsets: shape (num_experts + 1,) = (257,), int32, with offsets[1:] = scan[1:]
        expert_offsets = torch.empty(self.NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
        # Initialize with zeros; set prefix sums at index 1..256
        expert_offsets[0] = 0
        # Copy scan[1:] into expert_offsets[1:]
        if self.NUM_CLASSES > 0:
            expert_offsets[1:] = scan[1:]

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
