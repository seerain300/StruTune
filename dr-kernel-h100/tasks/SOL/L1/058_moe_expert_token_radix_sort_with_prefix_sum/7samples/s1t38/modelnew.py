import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements: tl.constexpr):
    """
    Build per-expert counts using atomic_add per element.
    flat_ptr: pointer to int32 flat array (length n_elements)
    counts_ptr: pointer to int32 counts array (length num_experts)
    n_elements: number of tokens
    """
    BLOCK_SIZE = 1024
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    for i in range(BLOCK_SIZE):
        idx = start + i
        if idx < n_elements:
            val = vals[i]
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts to produce expert offsets:
    offsets[i+1] = offsets[i] + counts[i], offsets[0] = 0.
    offsets_ptr: output int32 of length num_experts + 1
    counts_ptr: input int32 of length num_experts
    """
    total = 0
    for i in range(0, num_experts):
        ci = tl.load(counts_ptr + i)  # int32
        total += ci
        tl.store(offsets_ptr + i + 1, total)
    # offsets[0] is left as 0; we store the final total at last position, but since we only iterate up to num_experts-1,
    # we don't need to set offsets[0]. The caller can ensure offsets[0] == 0.
    # To be explicit, set offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # Ensure the last element equals total
    tl.store(offsets_ptr + num_experts, total)


@triton.jit
def _counting_sort_indices_kernel(flat_ptr, offsets_ptr, out_idx_ptr, n_elements: tl.constexpr):
    """
    Counting sort to produce sorted_token_indices:
    - For each i in [0, n_elements), read val = flat[i]
    - Emit i at position offsets[val], then increment offsets[val] += 1
    out_idx_ptr: output int32 permutation of length n_elements
    """
    for i in range(0, n_elements):
        val = tl.load(flat_ptr + i)  # int32
        pos = tl.load(offsets_ptr + val)  # int32
        tl.store(out_idx_ptr + pos, i)
        tl.atomic_add(offsets_ptr + val, 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute flat = topk_idx.reshape(-1) (int32)
        - Produce sorted_token_indices (int64) as permutation of 0..N-1 sorting by expert IDs (stable), using Triton counting sort.
        - Produce expert_offsets (int32) as inclusive prefix sums per expert using Triton.
        """
        # Ensure flat is on CUDA and int32
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        n = flat.numel()

        # 1) Triton histogram of counts
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        grid_counts = (triton.cdiv(n, 1024),)
        _histogram_counts_kernel[grid_counts](flat, counts, n_elements=n)

        # 2) Triton inclusive prefix sum to produce offsets (length num_experts + 1)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts=self.num_experts)

        # 3) Triton counting sort to produce sorted_token_indices (int64)
        out_idx = torch.empty(n, dtype=torch.int32, device=flat.device)
        # The counting sort kernel consumes 'offsets' as it emits elements; it will atomically update offsets on-the-fly.
        # Note: 'offsets' must start with the prefix sums for each expert; we need to copy 'counts' into 'offsets' for this.
        # We'll use a separate tensor for counting during sort. Create 'pos' as a temporary offsets for each expert's start.
        # However, to avoid extra passes, we recompute from 'counts' again using Triton scan. Since Triton only runs kernels,
        # we can just relaunch _inclusive_prefix_sum_kernel into a fresh buffer 'pos' initialized to zeros, and then use
        # 'offsets' for sorting by copying counts->pos and performing sort. To keep it minimal, we sort using counts directly
        # by building a temporary pos buffer. Triton kernels do not have access to Python variables outside, so we instead
        # perform the counting sort using the original offsets; but offsets are final cumsums. For counting sort, we need
        # the starting positions per expert. So we compute pos via Triton scan again.
        # Compute pos = inclusive scan of counts (same as offsets but stored into a new buffer). Then use pos for sorting.
        pos = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](counts, pos, num_experts=self.num_experts)

        # Now perform counting sort: we need offsets per expert to be pos for this phase. We will emulate by using pos as start
        # and then for each i, place it at pos[val] and increment pos[val]. To do this, we need a kernel that uses pos to write
        # and atomically increment per expert. Triton supports atomic_add, but we must avoid modifying counts during sort.
        # So we copy counts into offsets temporarily: set offsets = pos.copy().
        # However, Triton kernels do not have Python-side assignment to tensors, so we ensure 'offsets' points to pos before
        # sorting. To achieve this cleanly, we run the sorting kernel using pos as offsets for this phase only.
        # In practice, we can pass a different name; Triton kernels are pure functions and can read pos as offsets. We'll
        # proceed with sorting using pos as the current offsets buffer.

        # Launch counting sort kernel using pos as offsets for this phase. The kernel expects offsets_ptr to be the
        # per-expert starting positions. After each emit, we atomically increment pos[val] by 1. This produces sorted indices.
        _counting_sort_indices_kernel[()](
            flat, pos, out_idx, n_elements=n  # grid=(1,) would be fine; Triton will run it as a single program. To ensure it
                                             # runs with proper parallelism, we set grid to (1,) since this is a counting-sort
                                             # style kernel operating sequentially over i. Triton allows a single program to
                                             # perform this loop; the key is correctness.
        )

        sorted_token_indices = out_idx.to(torch.int64)
        return sorted_token_indices, offsets