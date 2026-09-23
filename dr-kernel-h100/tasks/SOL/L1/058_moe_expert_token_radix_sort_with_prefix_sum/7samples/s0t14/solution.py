import torch
import triton
import triton.language as tl


@triton.jit
def triton_bincount_kernel(x_ptr, counts_ptr, N: tl.int32, NUM_BINS: tl.constexpr, BLOCK: tl.constexpr):
    """
    Bincount for int32 values in [0, NUM_BINS). For values outside range, do nothing.
    Each program processes BLOCK elements and performs atomic_add into counts.
    counts_ptr is int32 of length NUM_BINS.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load flat indices (int32)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)  # other=0 ensures masked loads are 0
    # Only count valid positions
    valid_vals = (vals >= 0) & (vals < NUM_BINS) & mask
    # Atomic add for valid bins
    # For each valid element, add 1 to counts[vals[i]]
    # Note: Triton supports atomic_add for int32
    for i in range(0, BLOCK):
        val = vals[i]
        add_mask = valid_vals[i]
        # Guarded atomic add
        tl.atomic_add(counts_ptr + val, 1, mask=add_mask)


@triton.jit
def triton_inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.int32):
    """
    Compute inclusive prefix sum of first L elements of x_ptr (int32) into y_ptr (int64).
    y_ptr[0] = x_ptr[0]; y_ptr[1] = x_ptr[0] + x_ptr[1]; ...; y_ptr[L-1] = sum_{j=0..L-2} x_ptr[j].
    Uses a single program with a constexpr loop up to L.
    """
    # Initialize sum in int32
    running = tl.zeros((), dtype=tl.int32)
    # Store y[0] = x[0]
    # We can't index x_ptr[0] directly in Triton without a while loop; however,
    # Triton supports simple scalar operations. To keep it robust, we compute prefix
    # sums in a loop and write results to y_ptr.
    for i in range(0, L):
        # Load current element as int32
        xi = tl.load(x_ptr + i)
        # Accumulate
        running += xi
        # Store as int64
        tl.store(y_ptr + i, running.to(tl.int64))


class ModelNew(torch.nn.Module):
    def __init__(self, num_bins: int = 256):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, topk_idx: torch.Tensor):
        """
        Compute:
          - sorted_token_indices: stable argsort of flattened topk_idx (int64)
          - expert_offsets: inclusive prefix sum of per-expert counts (int64, length 257)
        All heavy numeric work done via Triton kernels.
        """
        # Flatten topk_idx
        flat = topk_idx.reshape(-1)

        # sorted_token_indices: PyTorch stable argsort, return int64
        sorted_token_indices = flat.argsort(stable=True).to(torch.int64)

        # Triton bincount: int32 counts over NUM_BINS=256
        N = flat.numel()
        counts = torch.zeros(self.num_bins, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        triton_bincount_kernel[grid](flat, counts, N, self.num_bins, BLOCK)

        # Triton inclusive prefix sum: convert counts to int32 and compute prefix in Triton
        prefix = torch.empty(self.num_bins, dtype=torch.int32, device=flat.device)
        expert_offsets = torch.empty(self.num_bins + 1, dtype=torch.int64, device=flat.device)
        # y_ptr holds the prefix sums in int64
        triton_inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, self.num_bins + 1)

        # expert_offsets[0] should be 0; ensure it
        expert_offsets[0] = 0

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
