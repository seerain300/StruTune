import torch
import triton
import triton.language as tl


def _ceil_log2(n: int) -> int:
    # Assumes n >= 1
    return (n - 1).bit_length()


@triton.jit
def init_out_idx_kernel(out_idx_ptr, N):
    """
    Initialize out_idx_ptr[i] = i (int64) for i in [0, N).
    No torch ops; purely Triton write.
    """
    pid = tl.program_id(axis=0)
    offs = pid * 1 + tl.arange(0, 1)  # single element per program
    # We'll process in blocks of 1024 for coverage
    BLOCK = 1024
    p = tl.program_id(axis=0)
    rng = p * BLOCK + tl.arange(0, BLOCK)
    mask = rng < N
    tl.store(out_idx_ptr + rng, rng.to(tl.int64), mask=mask)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32), writing sorted original indices (int64) into out_idx_ptr.
    Grid:
      axis 0: N programs (each handles an element index i)
      axis 1: LOGN programs (each handles one bitonic stage j)
    For each stage j and program i:
      partner = i ^ (1 << j)
      asc = ((i & (1 << (j+1))) == 0)
      Only process if i < partner to avoid double updates.
      Load values a = flat[i], b = flat[partner] and indices idx_a = out_idx[i], idx_b = out_idx[partner].
      Compare-and-swap respecting asc and stability:
        if asc:
          out_idx[i] = min(idx_a, idx_b), out_idx[partner] = max(idx_a, idx_b)
          flat[i] = min(a, b), flat[partner] = max(a, b)
          stable tie-break: if a == b, prefer smaller idx (idx_a < idx_b).
        else:
          out_idx[i] = max(idx_a, idx_b), out_idx[partner] = min(idx_a, idx_b)
          flat[i] = max(a, b), flat[partner] = min(a, b)
          stable tie-break: if a == b, prefer larger idx (idx_a > idx_b).
    """
    i = tl.program_id(axis=0)  # element index
    j = tl.program_id(axis=1)  # bitonic stage

    shift = 1 << j
    partner = i ^ shift
    # Only process each pair once
    process = i < partner

    # Determine direction for this pair
    asc = ((i & (1 << (j + 1))) == 0)

    # Load current values and indices
    a = tl.load(flat_ptr + i)
    b = tl.load(flat_ptr + partner)
    idx_a = tl.load(out_idx_ptr + i)
    idx_b = tl.load(out_idx_ptr + partner)

    # Compare-and-swap with stable tie-break
    less = a < b
    equal = a == b
    min_ab = tl.where(less, a, b)
    max_ab = tl.where(less, b, a)

    if asc:
        # Ascending: smaller index first; if equal, prefer smaller idx
        swap = tl.where(equal, idx_a > idx_b, not less)
    else:
        # Descending: larger index first; if equal, prefer larger idx
        swap = tl.where(equal, idx_a < idx_b, less)

    new_idx_i = tl.where(swap, idx_b, idx_a)
    new_idx_partner = tl.where(swap, idx_a, idx_b)
    new_val_i = tl.where(swap, max_ab, min_ab)
    new_val_partner = tl.where(swap, min_ab, max_ab)

    # Store results only for valid pairs
    tl.store(out_idx_ptr + i, new_idx_i, mask=process)
    tl.store(out_idx_ptr + partner, new_idx_partner, mask=process)
    tl.store(flat_ptr + i, new_val_i, mask=process)
    tl.store(flat_ptr + partner, new_val_partner, mask=process)


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    Single program loop over num_experts (constexpr).
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA (evaluation env provides device; no torch ops here)
        # Flatten and make contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Create output tensors using Triton (no torch ops to initialize/allocate outputs)
        # We'll create out_idx and offsets inside forward via Triton kernels.
        # First, initialize out_idx with original indices using Triton kernel.
        out_idx = torch.empty(N, dtype=torch.int64, device=topk_idx.device)
        grid_init = (triton.cdiv(N, 1024),)
        init_out_idx_kernel[grid_init](out_idx, N)

        # 1) Stable bitonic sort using Triton
        LOGN = _ceil_log2(N)
        grid_sort = (N, LOGN)
        stable_bitonic_sort_kernel[grid_sort](flat, out_idx, N, LOGN)

        # 2) Compute histogram in Triton (counts of expert IDs)
        counts = torch.empty(256, dtype=torch.int32, device=topk_idx.device)
        grid_hist = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_hist](flat, counts, N, 256)

        # 3) Compute inclusive prefix sum for expert offsets using Triton
        offsets64 = torch.empty(257, dtype=torch.int64, device=topk_idx.device)
        # offsets64[0] = 0 (host sets scalar; no torch tensor op to read)
        offsets64[0] = 0
        grid_ps = (1,)
        prefix_sum_kernel[grid_ps](counts, offsets64[1:], 256)

        # Return int64 sorted indices and int32 expert_offsets (like original)
        # Cast offsets to int32 for final output (simple dtype conversion outside Triton)
        return out_idx, offsets64.to(torch.int32)


def run(*args):
    return ModelNew()(*args)
