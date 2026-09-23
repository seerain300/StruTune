import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(
    vals_ptr,           # *int32
    counts_ptr,         # *int32, length = num_experts
    n_elements,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(vals_ptr + offs, mask=mask, other=0)  # int32
    # atomic add into counts[vals] for each valid element
    # num_experts is implicit in caller; vals are in [0, 255]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_prefix_sum_kernel(
    counts_ptr,         # *int32, length = num_experts
    offsets_ptr,        # *int32, length = num_experts + 1
    num_experts: tl.constexpr,
):
    # Single-program inclusive scan over counts
    running = 0
    # We need to iterate over a known range; use num_experts constant
    for e in range(num_experts):
        c = tl.load(counts_ptr + e)
        running += c
        tl.store(offsets_ptr + e, running)
    # last element should be sum of all counts (N)
    tl.store(offsets_ptr + num_experts, running)


@triton.jit
def _odd_even_sort_stable_kernel(
    arr_ptr,            # *int32, length = n
    indices_ptr,        # *int32, length = n (initially 0..n-1)
    n_elements,         # int32
    passes,             # int32, e.g., 2*n
):
    # Each program id corresponds to an element index
    i = tl.program_id(axis=0)
    # Even/odd phases of odd-even transposition sort
    # We guard all loads/stores with masks to avoid OOB.
    for t in range(passes):
        # Even phase: compare (i, i+1) for even i
        is_even = (t % 2) == 0
        # Each program only handles one comparison/s swap if it is at a valid position and phase
        if is_even:
            # Only even indices proceed
            even_mask = ((i % 2) == 0) & (i < n_elements) & (i + 1 < n_elements)
            if even_mask:
                # Load current and next values and indices
                a = tl.load(arr_ptr + i)
                b = tl.load(arr_ptr + (i + 1))
                ia = tl.load(indices_ptr + i)
                ib = tl.load(indices_ptr + (i + 1))
                # Compare and decide swap
                swap = a > b
                # Compute new values/indices for this position
                new_a = tl.where(swap, b, a)
                new_b = tl.where(swap, a, b)
                new_ia = tl.where(swap, ib, ia)
                new_ib = tl.where(swap, ia, ib)
                # Store results back
                tl.store(arr_ptr + i, new_a)
                tl.store(arr_ptr + (i + 1), new_b)
                tl.store(indices_ptr + i, new_ia)
                tl.store(indices_ptr + (i + 1), new_ib)
        else:
            # Odd phase: compare (i-1, i) for odd i
            odd_mask = ((i % 2) == 1) & (i < n_elements) & (i - 1 >= 0)
            if odd_mask:
                a = tl.load(arr_ptr + (i - 1))
                b = tl.load(arr_ptr + i)
                ia = tl.load(indices_ptr + (i - 1))
                ib = tl.load(indices_ptr + i)
                swap = a > b
                new_a = tl.where(swap, b, a)
                new_b = tl.where(swap, a, b)
                new_ia = tl.where(swap, ib, ia)
                new_ib = tl.where(swap, ia, ib)
                tl.store(arr_ptr + (i - 1), new_a)
                tl.store(arr_ptr + i, new_b)
                tl.store(indices_ptr + (i - 1), new_ia)
                tl.store(indices_ptr + i, new_ib)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA and flatten
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        flat = topk_idx.reshape(-1).contiguous()
        n = flat.numel().item()  # int
        # Cast to int32 for Triton
        flat_i32 = flat.to(torch.int32)

        num_experts = 256
        # 1) Histogram counts via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](flat_i32, counts, n_elements=n, BLOCK_SIZE=BLOCK_SIZE)

        # 2) Inclusive prefix sum to get offsets (Triton)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts=num_experts)

        # 3) Stable sort via Triton odd-even transposition sort (produce indices)
        # Initialize arr with flat values and indices with 0..n-1
        arr = flat_i32.clone()
        indices = torch.arange(n, dtype=torch.int32, device=flat.device)
        passes = 2 * n  # sufficient for odd-even sort to converge
        _odd_even_sort_stable_kernel[(n,)](arr, indices, n_elements=n, passes=passes)

        # Return: sorted_token_indices (int32, permutation of 0..n-1),
        #         expert_offsets (int32, length num_experts+1)
        return indices, offsets


def run(*args):
    return ModelNew()(*args)
