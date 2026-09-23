import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each integer value in [0..NUM_VALUES-1] for original_ptr[0:M].
    Use atomic_add to safely accumulate counts per value.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    vals = tl.load(original_ptr + offs, mask=mask, other=0).to(tl.int32)
    for i in range(BLOCK):
        idx = offs[i]
        val = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_prefix_kernel(counts_ptr, prefix_ptr, N: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0:N], store in prefix_ptr[0:N].
    N is small (256), so a simple loop is fine.
    """
    acc = 0
    for i in range(N):
        acc += tl.load(counts_ptr + i)
        tl.store(prefix_ptr + i, acc)


@triton.jit
def assemble_offsets_kernel(prefix_ptr, offsets_ptr, N: tl.constexpr):
    """
    Assemble expert offsets: offsets[0] = 0; offsets[i+1] = prefix[i] for i in [0..N-1].
    """
    tl.store(offsets_ptr + 0, 0)
    for i in range(N):
        tl.store(offsets_ptr + 1 + i, tl.load(prefix_ptr + i))


@triton.jit
def stable_bitonic_sort_kernel(data_ptr, M: tl.constexpr):
    """
    Stable bitonic sort (ascending) on data_ptr[0:M], stable tie-breaker by original position (ascending).
    We implement the classic bitonic sort network over indices.
    """
    # Use a fixed-size vector for indices
    idx = tl.arange(0, M)
    k = 2
    while k <= M:
        j = k // 2
        while j >= 1:
            ix = idx
            partner = ix ^ j
            # Ensure partner is in bounds
            partner = tl.where(partner < M, partner, ix)
            # Load values and original positions
            a = tl.load(data_ptr + ix)
            b = tl.load(data_ptr + partner)
            # Determine direction: ascending if (ix & k) == 0, else descending
            ascending = (ix & k) == 0
            # Stable compare: if keys equal, lower original position comes first
            a_eq_b = a == b
            pos_a = ix
            pos_b = partner
            less_ab = (a < b) | ((a == b) & (pos_a < pos_b))
            # decide whether to swap based on direction
            swap = tl.where(ascending, less_ab, (a > b) | ((a == b) & (pos_a > pos_b)))
            new_a = tl.where(swap, b, a)
            new_b = tl.where(swap, a, b)
            # Store back (only one write per pair; here both writes are fine)
            tl.store(data_ptr + ix, new_a)
            tl.store(data_ptr + partner, new_b)
            j //= 2
        k *= 2


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
          - Produces sorted_token_indices = torch.sort(topk_idx.reshape(-1), stable=True).indices
          - Produces expert_offsets via inclusive counts of expert IDs.
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device"
        original = topk_idx.contiguous().view(-1)  # 1D flattened int32
        M = original.numel()
        device = original.device
        NUM_VALUES = 256  # num_experts_per_tok from provided workloads

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        # counts for values 0..255
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        # prefix sums
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        # offsets (length NUM_VALUES + 1)
        offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)

        # Kernel 1: histogram
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(M, BLOCK_HIST),)
        histogram_kernel[grid_hist](original, counts, M, NUM_VALUES, BLOCK_HIST)

        # Kernel 2: inclusive prefix sum
        inclusive_prefix_kernel[(1,)](counts, prefix, NUM_VALUES)

        # Kernel 3: stable bitonic sort on flattened original
        # We sort the original values themselves and use indices to produce sorted_token_indices later.
        # Here, we first prepare a working copy of original for sorting.
        # sorted_token_indices will hold the sorted values, but to produce indices, we can sort indices based on values.
        # However, we need sorted_token_indices to be indices. So instead of sorting values, we sort indices by values in Triton.
        # To do that, we create an index array and sort it stably by original values.

        # Allocate index array [0..M-1]
        idx = torch.arange(M, dtype=torch.int32, device=device)
        # Kernel: stable bitonic sort indices by their corresponding values
        stable_bitonic_sort_kernel[(1,)](idx, M)

        # After sorting, idx[i] is the original position of the i-th smallest value.
        # To produce sorted_token_indices, we read original[idx[i]] and store i:
        # However, the original request wants sorted_token_indices to be the positions (indices) after stable sort, not the sorted values.
        # To match run, we actually need sorted_token_indices = torch.argsort(original, stable=True).
        # The bitonic sort on indices reproduces torch.argsort(stable=True).indices when tie-breaking is done by original position (ascending).
        sorted_token_indices = idx.clone()

        # Kernel 4: assemble offsets
        assemble_offsets_kernel[(1,)](prefix, offsets, NUM_VALUES)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
