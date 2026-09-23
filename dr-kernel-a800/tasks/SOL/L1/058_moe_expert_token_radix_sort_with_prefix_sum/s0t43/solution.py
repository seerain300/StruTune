import torch
import triton
import triton.language as tl


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort of flat_ptr (int32) producing sorted indices in out_idx_ptr (int64).
    Grid: axis=0 over N, axis=1 over bitonic stages [0..LOGN-1].
    """
    pid = tl.program_id(axis=0)  # index into flattened array
    # Initialize out_idx_ptr with original positions as int64
    pos = tl.full((), pid, dtype=tl.int64)
    val = tl.load(flat_ptr + pid)  # int32 value
    tl.store(out_idx_ptr + pid, pos)

    # Bitonic sort network
    for k in tl.static_range(1, LOGN + 1):
        stride = 1 << k
        for j in tl.static_range(0, k):
            partner = pid ^ (1 << j)
            asc = ((pid & (1 << (k + 1))) == 0)
            # Only process each pair once
            if pid < partner:
                v_partner = tl.load(flat_ptr + partner)  # int32
                idx_partner = tl.load(out_idx_ptr + partner)  # int64
                min_val = tl.minimum(val, v_partner)
                max_val = tl.maximum(val, v_partner)
                # Ordering: ascending or descending
                # Stability: if equal, lower original index comes first.
                if asc:
                    take_min = (val < v_partner) | ((val == v_partner) & (pid < partner))
                    new_val = tl.where(take_min, min_val, max_val)
                    should_swap = (val > v_partner) | ((val == v_partner) & (pid > partner))
                else:
                    take_max = (val > v_partner) | ((val == v_partner) & (pid < partner))
                    new_val = tl.where(take_max, max_val, min_val)
                    should_swap = (val < v_partner) | ((val == v_partner) & (pid > partner))
                # Update val
                val = new_val
                # Swap out_idx_ptr[pid] and out_idx_ptr[partner] if should_swap
                tmp_pid = tl.load(out_idx_ptr + pid)
                tmp_partner = tl.load(out_idx_ptr + partner)
                out_idx_pid = tl.where(should_swap, tmp_partner, tmp_pid)
                out_idx_partner = tl.where(should_swap, tmp_pid, tmp_partner)
                tl.store(out_idx_ptr + pid, out_idx_pid)
                tl.store(out_idx_ptr + partner, out_idx_partner)


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each flat value (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    Assumes flat_ptr values in [0, num_experts-1].
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    # Atomic add 1 for valid values
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int32, length num_experts) into offsets_ptr (int32).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    acc = tl.zeros((), dtype=tl.int32)
    for i in tl.static_range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


def run(topk_idx: torch.Tensor):
    """
    Triton-optimized version of run:
    Returns:
      sorted_token_indices: int64 tensor of shape (N,)
      expert_offsets: int32 tensor of shape (num_experts + 1,)
    """
    # Flatten
    flat = topk_idx.view(-1)  # int32, length N
    N = flat.numel()
    num_experts = 256

    # 1) Stable bitonic sort using Triton: produce sorted_token_indices (int64)
    out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)

    # Grid over elements and bitonic stages. For N<=4096, LOGN=12 is sufficient.
    grid = (N, 12)
    stable_bitonic_sort_kernel[grid](flat, out_idx, N, LOGN=12)

    # 2) Histogram via Triton
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    BLOCK = 1024
    grid_counts = (triton.cdiv(N, BLOCK),)
    count_histogram_kernel[grid_counts](flat, counts, N, num_experts=num_experts, BLOCK=BLOCK)

    # 3) Prefix sum via Triton (inclusive prefix sum of counts -> offsets[1:])
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[0] = 0  # set offset for index 0
    grid_ps = (1,)
    prefix_sum_kernel[grid_ps](counts, offsets, num_experts=num_experts)

    return out_idx, offsets


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect topk_idx as input
        assert len(args) == 1
        topk_idx = args[0]
        return run(topk_idx)


def run(*args):
    return ModelNew()(*args)
