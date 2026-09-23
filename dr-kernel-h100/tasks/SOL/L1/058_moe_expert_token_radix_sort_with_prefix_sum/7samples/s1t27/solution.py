import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(values_ptr, counts_ptr, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # Load flattened values as int32
    vals = tl.load(values_ptr + offs, mask=mask, other=0)
    vals = vals.to(tl.int32)
    # Perform atomic add per element into counts
    # Note: counts array length is num_experts (compile-time constexpr? We pass dynamic length, so use index as int32)
    # Triton supports atomic add to pointers; here we index by vals (which are in range [0, num_experts-1]).
    # We need to cast index to int32 pointer arithmetic supported by Triton for atomic add.
    # counts_ptr is a pointer to int32; vals are int32 indices.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    # Single program performs inclusive scan: offsets[i] = sum_{j < i} counts[j]
    # offsets_ptr has length num_experts + 1
    # We'll loop i from 0 to num_experts-1. offsets[0] = 0, offsets[1] = counts[0], etc.
    # This is fine since num_experts is small (256).
    # Initialize offsets[0] = 0
    # Triton supports scalar operations in a single program. We do a simple loop.
    acc = tl.zeros((), dtype=tl.int32)
    # We need a Python-like for loop: for i in range(num_experts):
    # Triton allows such loops; acc is scalar, counts_ptr[i] is load; offsets_ptr[i+1] = acc
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)
    # offsets_ptr[0] remains 0; we set it explicitly
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))


@triton.jit
def _odd_even_sort_indices_kernel(arr_ptr, indices_ptr, n_elements: tl.int32, passes: tl.int32):
    # Each program owns one position i in the permutation.
    # It performs compare-swap with neighbor during even/odd phases.
    i = tl.program_id(0)
    # bounds check
    valid = i < n_elements
    # We'll run a fixed number of passes. Each pass:
    # even phase: i even compare with i+1; odd phase: i odd compare with i-1.
    # We guard updates by checking i+1 < n_elements and only lanes with even/odd i perform the compare-swap.
    # We maintain arr_ptr[i] and indices_ptr[i]. Since all programs update, we rely on masked loads/stores.
    for t in range(0, passes):
        # Even phase
        if (i % 2 == 0) & valid:
            j = i + 1
            j_valid = j < n_elements
            val_i = tl.load(arr_ptr + i)
            val_j = tl.load(arr_ptr + j, mask=j_valid, other=val_i)
            idx_i = tl.load(indices_ptr + i)
            idx_j = tl.load(indices_ptr + j, mask=j_valid, other=idx_i)
            swap = val_i > val_j
            new_val_i = tl.where(swap, val_j, val_i)
            new_val_j = tl.where(swap, val_i, val_j)
            new_idx_i = tl.where(swap, idx_j, idx_i)
            new_idx_j = tl.where(swap, idx_i, idx_j)
            tl.store(arr_ptr + i, new_val_i, mask=valid)
            tl.store(arr_ptr + j, new_val_j, mask=j_valid)
            tl.store(indices_ptr + i, new_idx_i, mask=valid)
            tl.store(indices_ptr + j, new_idx_j, mask=j_valid)
        # Odd phase
        if (i % 2 == 1) & valid:
            j = i - 1
            j_valid = j >= 0
            val_i = tl.load(arr_ptr + i)
            val_j = tl.load(arr_ptr + j, mask=j_valid, other=val_i)
            idx_i = tl.load(indices_ptr + i)
            idx_j = tl.load(indices_ptr + j, mask=j_valid, other=idx_i)
            swap = val_i > val_j
            new_val_i = tl.where(swap, val_j, val_i)
            new_val_j = tl.where(swap, val_i, val_j)
            new_idx_i = tl.where(swap, idx_j, idx_i)
            new_idx_j = tl.where(swap, idx_i, idx_j)
            tl.store(arr_ptr + i, new_val_i, mask=valid)
            tl.store(arr_ptr + j, new_val_j, mask=j_valid)
            tl.store(indices_ptr + i, new_idx_i, mask=valid)
            tl.store(indices_ptr + j, new_idx_j, mask=j_valid)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Ensure we're on CUDA
        if not topk_idx.is_cuda:
            # Move to CUDA if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            topk_idx = topk_idx.to(device)

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        n_elements = flat.numel()

        # Allocate outputs
        # sorted_token_indices: permutation of 0..N-1 (int32)
        indices = torch.arange(n_elements, dtype=torch.int32, device=flat.device)

        # arr: copy of flat values for sorting in Triton
        arr = flat.to(torch.int32).contiguous()

        # counts per expert
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        # offsets (inclusive prefix sum)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel: one pass over N elements
        grid_histogram = (triton.cdiv(n_elements, 1024),)  # BLOCK_SIZE = 1024
        _histogram_counts_kernel[grid_histogram](
            flat.to(torch.int32).contiguous(),  # values_ptr
            counts,                            # counts_ptr
            n_elements,                        # n_elements
            BLOCK_SIZE=1024,
            num_warps=4,
        )

        # Compute inclusive prefix sum in Triton (single program instance)
        # num_experts is known at module init
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # Launch Triton odd-even sort to produce sorted_token_indices
        # We perform 2*N passes for robust sorting
        passes = 2 * n_elements
        _odd_even_sort_indices_kernel[(n_elements,)](
            arr, indices, n_elements, passes, num_warps=1
        )

        # Return sorted_token_indices and expert_offsets
        # The original returns int32 for indices and int32 offsets.
        return indices, offsets


def run(*args):
    return ModelNew()(*args)
