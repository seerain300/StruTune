import torch
import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of values in flat_ptr (int32), one pass with atomic adds.
    flat_ptr: *int32, shape [N]
    counts_ptr: *int32, shape [num_experts]
    N: int, total number of elements
    Each program processes BLOCK elements; masked loads prevent OOB.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    # Scalar loop per element for robust masking
    for i in range(BLOCK):
        idx = start + i
        m = mask[i]
        if m:
            id_val = tl.load(flat_ptr + idx)
            tl.atomic_add(counts_ptr + id_val, 1)


@triton.jit
def compute_le_counts(counts_ptr, le_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr into le_ptr for k in [0, num_experts-1].
    counts_ptr: *int32, shape [num_experts]
    le_ptr: *int32, shape [num_experts]
    """
    running = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_experts):
        c = tl.load(counts_ptr + k)
        running += c
        tl.store(le_ptr + k, running)


@triton.jit
def compute_lt_counts(le_ptr, counts_ptr, lt_ptr, num_experts: tl.int32):
    """
    Compute lt_counts[k] = le_counts[k] - counts[k], exclusive less-than.
    le_ptr: *int32, shape [num_experts]
    counts_ptr: *int32, shape [num_experts]
    lt_ptr: *int32, shape [num_experts]
    """
    for k in range(0, num_experts):
        le_k = tl.load(le_ptr + k)
        cnt_k = tl.load(counts_ptr + k)
        tl.store(lt_ptr + k, le_k - cnt_k)


@triton.jit
def compute_out_pos(flat_ptr, out_ptr, le_ptr, lt_ptr, N, num_experts: tl.int32):
    """
    Stable argsort permutation:
    - For each i in [0, N), id = flat[i]
    - pos = le[id] - (1 if id has duplicates else 0), ensuring ties are ordered by i (stable).
    - Write i (original position) into out[pos].
    We process elements in blocks of BLOCK, guarded by mask.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 256  # fixed block for simplicity; ensure N divisible or mask handles it
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    for i in range(BLOCK):
        idx = start + i
        m = mask[i]
        if m:
            id_val = tl.load(flat_ptr + idx)
            le_val = tl.load(le_ptr + id_val)
            lt_val = tl.load(lt_ptr + id_val)
            # Stable tie-break: if duplicates exist for this id, subtract 1 so earlier index comes first
            has_duplicate = lt_val < le_val
            pos = le_val - tl.where(has_duplicate, 1, 0)
            # Scatter original index (idx) into output at position 'pos'
            tl.store(out_ptr + pos, idx)


@triton.jit
def prefix_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr into offsets_ptr[1:], with offsets_ptr[0] = 0.
    counts_ptr: *int32, shape [num_experts]
    offsets_ptr: *int32, shape [num_experts+1]
    """
    running = tl.zeros((), dtype=tl.int32)
    # Initialize first offset to 0
    tl.store(offsets_ptr + 0, running)
    for k in range(0, num_experts):
        c = tl.load(counts_ptr + k)
        running += c
        tl.store(offsets_ptr + 1 + k, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure on CUDA for Triton
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        # Flatten and ensure int32
        flat = topk_idx.reshape(-1)
        assert flat.dtype == torch.int32, "flat must be int32"
        N = flat.numel()
        num_experts = 256  # same as original code

        # 1) Histogram via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK_HIST)

        # 2) Compute le_counts (inclusive) via Triton
        le_counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        compute_le_counts[(1,)](counts, le_counts, num_experts)

        # 3) Compute lt_counts (exclusive less-than) via Triton
        lt_counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        compute_lt_counts[(1,)](le_counts, counts, lt_counts, num_experts)

        # 4) Compute stable argsort permutation via Triton and write output
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        grid_argsort = (triton.cdiv(N, 256),)
        compute_out_pos[grid_argsort](flat, sorted_token_indices, le_counts, lt_counts, N, num_experts)

        # 5) Compute expert_offsets via Triton prefix scan
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        prefix_scan_kernel[(1,)](counts, expert_offsets, num_experts)

        # Return results
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
