import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    # Returns the smallest power of two >= n
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). out_idx_ptr holds int64 original positions 0..N-1.
    For each bitonic stage j, each program i computes partner = i ^ (1 << j).
    Ascending if (i & (1 << (j+1))) == 0 else descending. Stability: tie-break by i < partner.
    Only process pairs with i < partner to avoid races.
    """
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Initialize out_idx_ptr with original positions
    tl.store(out_idx_ptr + pid, tl.full((), pid, tl.int64))

    # Bitonic sort stages
    for k in range(1, LOGN):  # k = bit length of comparator
        for j in range(k - 1, -1, -1):  # j is the stage within bit length k
            partner = pid ^ (1 << j)
            # Only process each pair once
            if pid < partner:
                a = tl.load(flat_ptr + pid)
                b = tl.load(flat_ptr + partner)
                ia = tl.load(out_idx_ptr + pid)
                ib = tl.load(out_idx_ptr + partner)

                # Determine direction
                asc = ((pid & (1 << (j + 1))) == 0)

                # Compare and apply stable sort tie-break
                eq = (a == b)
                less = (a < b)
                # Stable tie-break: if equal, lower original index should come first
                lower_prefers_first = (eq & (pid < partner))
                higher_prefers_first = (eq & (partner < pid))

                # If ascending: swap when (a > b) or (equal and pid > partner)
                # If descending: swap when (a < b) or (equal and pid < partner)
                swap = ( (asc and ((a > b) or (lower_prefers_first))) or
                         ((not asc) and ((a < b) or (higher_prefers_first))) )

                if swap:
                    # Swap indices only; we don't need to return sorted flat values
                    tl.store(out_idx_ptr + pid, ib)
                    tl.store(out_idx_ptr + partner, ia)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure on CUDA and contiguous
        device = topk_idx.device
        assert device.type == "cuda", "ModelNew expects CUDA tensors for Triton kernels."

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        LOGN = _next_power_of_2(N)
        num_experts = 256

        # Output tensors
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=device)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[0] = 0  # placeholder; will be overwritten by prefix sum for positions 1.. after kernel

        # Histogram counts (int32)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        count_histogram_kernel[(triton.cdiv(N, 1024),)](flat, counts, N, num_experts)

        # Inclusive prefix sum of counts -> offsets (int64 in kernel, convert to int32 after)
        offsets64 = torch.empty(num_experts, dtype=torch.int64, device=device)
        prefix_sum_kernel[(num_experts,)](counts, offsets64, num_experts)
        expert_offsets[1:] = offsets64.to(torch.int32)

        # Stable sort indices via Triton
        work_flat = flat.contiguous()
        work_sorted_idx = sorted_token_indices
        stable_bitonic_sort_kernel[(N,)](work_flat, work_sorted_idx, N, LOGN)

        return work_sorted_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
