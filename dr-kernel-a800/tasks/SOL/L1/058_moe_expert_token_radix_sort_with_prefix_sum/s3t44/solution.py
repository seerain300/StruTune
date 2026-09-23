import torch
import triton
import triton.language as tl


@triton.jit
def stable_sort_indices_by_values(flat_ptr, sorted_idx_ptr, N):
    """
    Sort 'flat_ptr' in ascending order stably and write the original indices into 'sorted_idx_ptr'.
    For equal values, smaller original index comes first (stable=True).
    """
    idx = torch.arange(0, N, dtype=tl.int32, device=flat_ptr.device)
    # Copy original indices to output
    for i in range(N):
        tl.store(sorted_idx_ptr + i, idx[i])

    # Insertion sort with stability handling: scan from left and insert flat[i] into sorted_idx
    # Compare-exchange with tie-break by original index (ascending index for equals).
    for i in range(1, N):
        v = tl.load(flat_ptr + i)
        pos = i
        # Move v to its position by shifting elements to the right as needed
        for j in range(i - 1, -1, -1):
            vj = tl.load(flat_ptr + j)
            ii = tl.load(sorted_idx_ptr + j)
            # If vj > v or (vj == v and j > pos), shift
            need_shift = (vj > v) | ((vj == v) & (j > i))
            # Shift: flat[j+1] = vj, sorted_idx[j+1] = ii
            # We emulate shift by storing into positions j+1
            # Note: this is a simple per-iteration approach; Triton will compile the loop.
            # When need_shift, we store vj at j+1 and ii at sorted_idx[j+1].
            if need_shift:
                tl.store(flat_ptr + (j + 1), vj)
                tl.store(sorted_idx_ptr + (j + 1), ii)
        # After loop, place v at position 'pos'
        tl.store(flat_ptr + i, v)
        tl.store(sorted_idx_ptr + i, tl.int32(i))


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_bins: tl.constexpr):
    """
    Count occurrences of each integer value in flat_ptr into counts_ptr[0:num_bins].
    Assumes flat_ptr values are in [0, num_bins-1]. We add +1 for each occurrence.
    """
    # Zero-initialize counts (forward caller should ensure counts initialized to zero).
    for b in range(num_bins):
        tl.store(counts_ptr + b, tl.zeros((), dtype=tl.int32))
    # Scan flat and increment counts
    for i in range(N):
        val = tl.load(flat_ptr + i)
        # increment counts[val]
        current = tl.load(counts_ptr + val)
        current += 1
        tl.store(counts_ptr + val, current)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sums of counts_ptr into offsets_ptr[0:N_bins].
    offsets[0] = 0; offsets[i] = sum_{k=0..i-1} counts[k], for i >= 1.
    """
    # Initialize offsets[0] = 0
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure device is CUDA for Triton
        device = topk_idx.device
        # Flatten
        flat = topk_idx.reshape(-1).contiguous()

        # 1) Stable sort via Triton
        N = flat.numel()
        sorted_idx = torch.empty(N, dtype=torch.int32, device=device)
        grid = (1,)
        stable_sort_indices_by_values[grid](flat, sorted_idx, N, num_warps=1)

        # 2) Triton histogram for bincount
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        # Note: Triton loop over N uses dynamic range; num_bins is constexpr for unrolling
        count_histogram_kernel[grid](flat, counts, N, num_bins=self.num_experts, num_warps=1)

        # 3) Triton exclusive prefix sum to produce expert_offsets
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        exclusive_prefix_sum_kernel[grid](counts, expert_offsets, N_bins=self.num_experts, num_warps=1)

        return sorted_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
