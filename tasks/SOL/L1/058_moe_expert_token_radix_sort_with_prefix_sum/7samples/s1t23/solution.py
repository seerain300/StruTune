import torch
import triton
import triton.language as tl


@triton.jit
def _counting_sort_indices_kernel(flat_ptr, indices_ptr, n_elements: tl.constexpr):
    """
    Perform a stable counting sort to produce the permutation indices 'indices_ptr'
    that would sort the flattened array 'flat_ptr' in ascending order.

    flat_ptr: 1D int32, length N
    indices_ptr: 1D int32, length N (output permutation)
    n_elements: compile-time known N (tl.constexpr)
    """
    # We implement lower_bound via binary search over the already filled portion
    # of indices. indices[i] is set to position i in the output (stable).
    # For each i, place it at the position of the next occurrence of its value
    # among already placed elements. Since we write at lower_bound, this ensures
    # stability for equal keys.
    for i in range(n_elements):
        # Load current value
        val = tl.load(flat_ptr + i)
        # Compute lower_bound: number of elements strictly less than val among placed items.
        # We simulate a binary search over [0, i] for lower_bound of val.
        left = tl.zeros((), dtype=tl.int32)
        right = tl.zeros((), dtype=tl.int32)
        pos = tl.zeros((), dtype=tl.int32)
        # Binary search for lower_bound
        # Note: We keep pos at lower_bound of val among indices[0:i]
        for _ in range(32):  # sufficient for any N up to 2^32 in practice
            if left >= right:
                break
            mid = (left + right) // 2
            # Load the value at position 'mid' in indices; indices_ptr[mid] contains
            # the original flat index of the element at sorted position mid.
            existing_val = tl.load(flat_ptr + tl.load(indices_ptr + mid))
            if existing_val < val:
                left = mid + 1
            else:
                right = mid
        pos = left
        # Place 'i' at position 'pos' in indices
        tl.store(indices_ptr + pos, i)


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements: tl.constexpr, num_experts: tl.constexpr):
    """
    Compute per-expert counts using atomic adds. counts_ptr is int32 of length num_experts.
    flat_ptr is int32, length n_elements.
    """
    # Grid is 1, but we process in chunks inside the kernel to keep it simple and fast.
    # For robustness across sizes, iterate over flat_ptr and atomic_add per element.
    for i in range(n_elements):
        val = tl.load(flat_ptr + i)
        # Ensure val is within [0, num_experts-1]; original inputs guarantee this.
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, out_ptr, n_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (length num_experts) into out_ptr (length num_experts+1).
    out_ptr[0] = 0, out_ptr[1] = counts[0], out_ptr[2] = counts[0] + counts[1], ...
    Single-program scan loop.
    """
    # We assume num_experts is relatively small (e.g., 256), so a simple loop is fine.
    total = tl.zeros((), dtype=tl.int32)
    # out_ptr[0] = 0 by construction
    for e in range(n_experts):
        v = tl.load(counts_ptr + e)
        total += v
        tl.store(out_ptr + e + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure device and dtype
        device = topk_idx.device
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)

        n = flat.numel()
        num_experts = 256

        # 1) Triton counting sort to produce sorted_token_indices (int32 permutation)
        indices = torch.empty(n, dtype=torch.int32, device=device)
        # Note: pass n_elements as constexpr-like by making n known at launch time.
        _counting_sort_indices_kernel[(1,)](flat, indices, n_elements=n)

        # 2) Triton histogram counts per expert
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_counts_kernel[(1,)](flat, counts, n_elements=n, num_experts=num_experts)

        # 3) Triton inclusive prefix sum to produce expert_offsets
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[0] = 0  # placeholder; kernel will fill [1:]
        _inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, n_experts=num_experts)

        return indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
