import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements, num_experts: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Build per-expert counts for values in flat_ptr.
    For each element in flat_ptr:
      if 0 <= value < num_experts: counts[value] += 1
    flat_ptr: int32, shape [n_elements]
    counts_ptr: int32, shape [num_experts]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a block of values
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32

    # Accumulate into counts via atomics
    for i in range(BLOCK_SIZE):
        idx = offsets[i]
        m = mask[i]
        val = vals[i]
        # Only increment if within range
        cond = m & (val >= 0) & (val < num_experts)
        # Atomic add to counts[val]
        tl.atomic_add(counts_ptr + val, 1, mask=cond)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0:num_experts] into offsets_ptr[0:num_experts].
    Set offsets_ptr[num_experts] = N (number of tokens).
    counts_ptr: int32, length num_experts
    offsets_ptr: int32, length num_experts + 1
    """
    # We can do a sequential loop inside one program instance
    total = 0
    # For i in 0..num_experts-1: offsets[i] = total; total += counts[i]
    for i in range(num_experts):
        total += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i, total)
    # offsets[num_experts] = N (global tokens count). We don't have N here; host must precompute total or pass it.
    # We'll leave offsets[num_experts] as zero and set in host after this kernel; or we can write total one more time.
    # But since this is Triton-only for correctness, we'll assume host will set the last element after kernel runs.
    # To ensure correctness, we'll compute total and store it into offsets_ptr[num_experts] using a global N passed.
    # However, Triton kernels can't directly access Python variables. So we precompute N on host and pass it in offsets_ptr[num_experts].
    # Since we cannot modify offsets_ptr[num_experts] from here, host will handle setting it. The code below is fine:
    pass  # We'll set offsets[num_experts] = N in host code after the kernel.


@triton.jit
def _odd_even_transpose_sort_indices_kernel(arr_ptr, indices_ptr, n_elements, n_passes: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Odd-even transposition sort on arr_ptr (int32) of length n_elements.
    Track permutation in indices_ptr (int32), initialized to 0..n_elements-1.
    For each pass t in [0, n_passes):
      if t % 2 == 0 (even phase): i even compare-swap with i+1
      else (odd phase): i odd  compare-swap with i-1
    Only perform when indices are in range. Stable: equal values not swapped.
    arr_ptr: int32, length n_elements (global buffer, read/write)
    indices_ptr: int32, length n_elements (permutation buffer)
    """
    # Note: This kernel assumes arr_ptr and indices_ptr are global arrays accessible across all programs.
    # Each program handles one index element and performs compare-swap with its partner in the current phase.
    # Launch grid = (n_elements,)
    pid = tl.program_id(0)
    i = pid

    # Even phase: i even compares with i+1; Odd phase: i odd compares with i-1
    # We loop over passes and update arr and indices accordingly.
    # We load current values using indices; then update arr at those positions.
    for _ in range(n_passes):
        # Determine phase
        phase = _  % 2  # 0: even, 1: odd
        # For even phase, i even; for odd phase, i odd
        # Compute partner and load current values and indices
        if phase == 0:
            # Even phase
            # Only process even i
            if (i % 2 == 0) & (i + 1 < n_elements):
                # Load current values at positions arr[i] and arr[i+1]
                a_i = tl.load(arr_ptr + i)
                a_j = tl.load(arr_ptr + (i + 1))
                # Indices of these positions
                idx_i = tl.load(indices_ptr + i)
                idx_j = tl.load(indices_ptr + (i + 1))
                # Stable: only swap if a_i > a_j
                if a_i > a_j:
                    # Swap values in arr using indices
                    tl.store(arr_ptr + idx_i, a_j)
                    tl.store(arr_ptr + idx_j, a_i)
                    # Update indices to reflect the swap (indices are swapped accordingly)
                    # We need to update indices[i] and indices[i+1] to new positions, but since we use direct stores via idx_i, idx_j,
                    # indices already point to those positions, so nothing extra needed here for indices.
        else:
            # Odd phase
            # Only process odd i
            if (i % 2 == 1) & (i - 1 >= 0):
                a_i = tl.load(arr_ptr + i)
                a_j = tl.load(arr_ptr + (i - 1))
                idx_i = tl.load(indices_ptr + i)
                idx_j = tl.load(indices_ptr + (i - 1))
                if a_i > a_j:
                    tl.store(arr_ptr + idx_i, a_j)
                    tl.store(arr_ptr + idx_j, a_i)
        # After each pass, arr_ptr and indices_ptr reflect the current state. The loops run many times to converge.

# In ModelNew.forward, we'll:
# 1) Flatten topk_idx to flat (int32, 1D), ensure contiguity.
# 2) Allocate counts (int32, length num_experts) to zeros.
# 3) Launch _histogram_counts_kernel on flat.
# 4) Allocate offsets (int32, length num_experts+1) to zeros.
# 5) Launch _inclusive_prefix_sum_kernel(counts, offsets); then set offsets[-1] = N on host.
# 6) Initialize arr = flat.clone().contiguous() and indices = torch.arange(N, device=flat.device, dtype=torch.int32).
# 7) Launch _odd_even_transpose_sort_indices_kernel(arr, indices, N, n_passes=100, BLOCK_SIZE=256).
# 8) Return indices (sorted_token_indices) and offsets (expert_offsets).
# Note: For correctness, we need to ensure _inclusive_prefix_sum_kernel sets offsets[-1] = N. Triton kernel above doesn't, so we do it in host.
# Also, Triton kernels operate on device tensors; we don't perform torch.sort or torch.bincount in host code.

class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure device and dtype
        if not topk_idx.is_cuda:
            # If not on CUDA, move to CUDA to run Triton. Evaluation harness typically provides CUDA tensors.
            topk_idx = topk_idx.cuda()
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)  # 1D int32
        n = flat.numel()
        num_experts = 256  # fixed as per original code

        # 1) Histogram with Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        grid_hist = (triton.cdiv(n, 1024),)
        _histogram_counts_kernel[grid_hist](flat, counts, n, num_experts, BLOCK_SIZE=1024)

        # 2) Inclusive prefix sum with Triton; then set last element to N on host
        offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)  # single program instance
        # Set offsets[-1] = N (total tokens)
        offsets[-1] = n

        # 3) Prepare for sorting
        # Copy flat to arr (int32), initialize indices buffer
        arr = flat.clone().contiguous().to(torch.int32)
        indices = torch.arange(n, device=flat.device, dtype=torch.int32)

        # 4) Triton odd-even transposition sort for indices
        grid_sort = (n,)
        _odd_even_transpose_sort_indices_kernel[grid_sort](arr, indices, n, n_passes=100, BLOCK_SIZE=256)

        # Return sorted indices and offsets
        # Note: original returns sorted_token_indices (int32) and expert_offsets (int32).
        return indices, offsets