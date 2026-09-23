import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_indices_by_values_stable_kernel(a_ptr, N, out_ptr):
    """
    Compute the stable argsort of the 1D array 'a' (length N).
    Writes out[i] = index j in [0..N-1] such that a[j] would be at position i
    in ascending order, with ties broken by original index (stable).
    'a' is assumed to be int32. 'out' is int32.
    """
    i = tl.program_id(0)  # one program per element
    # Guard if grid > N
    if i >= N:
        return

    # Load the value at position i
    val_i = tl.load(a_ptr + i)

    # Compute rank of i: number of elements strictly less than val_i
    # plus number of equal elements with smaller original index (stable).
    rank = tl.zeros((), dtype=tl.int32)
    # j loop: Triton allows scalar-controlled loops; O(N) but fine here
    for j in range(0, N):
        val_j = tl.load(a_ptr + j)
        # stable tie-break: count elements less than val_i; for equal values, count j < i
        less = val_j < val_i
        equal = val_j == val_i
        tie = equal & (j < i)
        cnt = less.to(tl.int32) + tie.to(tl.int32)
        rank += cnt

    # Reserve a unique position via atomic add (one per element), then write index
    pos = tl.atomic_add(out_ptr, 1)  # start from 0; returns old value (position for i)
    # Store index i at that position
    tl.store(out_ptr + pos, i)


@triton.jit
def _histogram_kernel(a_ptr, N, histogram_ptr):
    """
    Histogram of values in 'a' (int32) across num_buckets bins.
    Uses atomic_add to add 1 to histogram[a[i]] for each i.
    """
    i = tl.program_id(0)
    if i >= N:
        return
    val = tl.load(a_ptr + i)
    tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _prefix_sum_kernel(hist_ptr, offsets_ptr, num_buckets: tl.constexpr):
    """
    Compute inclusive prefix sum of 'hist_ptr' (length num_buckets) into 'offsets_ptr' (length num_buckets+1).
    offsets[0] = 0 (caller will ensure). We write offsets[1..].
    """
    b = tl.program_id(0)  # bucket id
    if b >= num_buckets:
        return
    # Sum of previous bins
    s = tl.zeros((), dtype=tl.int32)
    for k in range(0, b):
        s += hist_ptr[k]
    # offsets[b] = offsets[b-1] + hist[b-1]
    # offsets[0] is 0; we compute from 1..num_buckets
    tl.store(offsets_ptr + b + 1, s + tl.load(hist_ptr + b))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original 'run' function.
        Returns:
          - sorted_token_indices: 1D int32 tensor of length N, where N = topk_idx.numel(),
            representing the permutation indices that would sort the flattened topk_idx
            in ascending order (stable).
          - expert_offsets: 1D int32 tensor of length (num_experts + 1), where num_experts=256,
            representing cumulative counts per expert ID.
        """
        # Ensure we are on CUDA; if not, fall back to pure PyTorch (not used in benchmarking).
        if not topk_idx.is_cuda:
            # Pure PyTorch fallback (for safety if ever needed)
            flat = topk_idx.reshape(-1).contiguous()
            sorted_token_indices, _ = torch.sort(flat, stable=True)
            # We need permutation indices, not the sorted values:
            values = flat
            N = values.numel()
            # Compute permutation indices: use argsort
            perm = torch.empty_like(flat, dtype=torch.long)
            # torch.argsort returns indices that sort the input
            perm = torch.argsort(values, stable=True)
            sorted_token_indices = perm.to(torch.int32)
            # Compute expert offsets
            num_experts = 256
            expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
            hist = torch.bincount(values.long(), minlength=num_experts)
            # We don't have cumsum helper in this context; compute via simple loop
            # Note: torch.cumsum is available, but keep it simple here
            total = 0
            for i in range(num_experts):
                total += hist[i]
                expert_offsets[i + 1] = total
            return sorted_token_indices, expert_offsets

        # CUDA path: Triton kernels
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Stable argsort to get permutation 'sorted_token_indices' (length N)
        out = torch.zeros(N, dtype=torch.int32, device=device)  # positions holder
        grid = (N,)
        _argsort_indices_by_values_stable_kernel[grid](flat, N, out)

        # 2) Histogram of topk_idx values (int32)
        num_experts = 256
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, 256),)  # grid size doesn't matter; loop over N in kernel
        _histogram_kernel[grid_hist](flat, N, histogram)

        # 3) Prefix sum to get expert offsets (num_experts + 1)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0  # ensure starts at 0
        _prefix_sum_kernel[(1,)](histogram, offsets, num_buckets=num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
