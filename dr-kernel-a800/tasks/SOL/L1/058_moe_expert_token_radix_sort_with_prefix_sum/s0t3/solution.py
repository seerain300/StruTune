import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_sort_stable_indices_1d(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32) and produce sorted indices in out_idx_ptr (int64).
    Grid: axis=0 has size N (one program per element), axis=1 has size LOGN (one stage per k).
    For each stage j, each program with lane i computes partner = i ^ (1 << j).
    It performs a single pairwise compare-and-swap with partner if i < partner.
    Stability: for equal values, the smaller original index (i < partner) comes first.
    """
    axis0 = tl.program_id(axis=0)  # lane i
    axis1 = tl.program_id(axis=1)  # stage j

    # In each stage j, k is the bit position for bitonic sequence
    k = axis1  # 0..LOGN-1

    # Number of elements in this bitonic sequence stage is 1 << (k+1)
    step = 1 << k
    partner = axis0 ^ step

    # Only process each pair once and within bounds
    do_pair = axis0 < partner
    in_bounds = (axis0 < N) & (partner < N) & do_pair

    # Load current values
    a_val = tl.load(flat_ptr + axis0, mask=in_bounds, other=0)  # int32 value at i
    b_val = tl.load(flat_ptr + partner, mask=in_bounds, other=0)  # int32 value at partner

    # Ascending/descending for this stage: elements where (axis0 & (1<<k)) == 0 go ascending, else descending
    asc = ((axis0 & (1 << k)) == 0)

    # Determine if we should take a_val or b_val for position i
    # If ascending: take a if a <= b (stable tie-break: i < partner when equal)
    # If descending: take b if b <= a (stable tie-break: partner < i when equal)
    take_a = tl.where(asc, (a_val <= b_val), (b_val <= a_val))
    # Stable tie-break: if equal, prefer smaller index
    tie_a = (a_val == b_val) & (axis0 < partner)
    take_a = take_a | tie_a

    new_i_val = tl.where(take_a, a_val, b_val)
    new_p_val = tl.where(take_a, b_val, a_val)

    # Store back to out_idx_ptr as int64 indices
    # Note: we assume out_idx_ptr is int64 and flat_ptr is int32
    tl.store(out_idx_ptr + axis0, new_i_val.to(tl.int64), mask=in_bounds)
    tl.store(out_idx_ptr + partner, new_p_val.to(tl.int64), mask=in_bounds)


@triton.jit
def count_histogram_atomic_kernel(flat_ptr, offsets_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute histogram of flat indices and atomically accumulate into offsets[1:].
    offsets has length num_experts + 1, dtype int64.
    For each element flat[i] in [0, num_experts-1], increment offsets[flat[i] + 1].
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values
    # Atomic add 1 (int64) into offsets[val + 1]
    tl.atomic_add(offsets_ptr + (vals + 1), 1, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - sorted_token_indices: computed via Triton stable bitonic sort (int64)
        - expert_offsets: computed via Triton histogram (atomic adds) + cumsum in PyTorch, then cast to int32
        Returns:
            sorted_token_indices: torch.Tensor (int64) shape (N,)
            expert_offsets: torch.Tensor (int32) shape (num_experts + 1,)
        """
        # Ensure input on CUDA and int32
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # 1) Stable sort via Triton bitonic sort; produce int64 indices
        # Determine LOGN for bitonic sort
        LOGN = (N - 1).bit_length() if N > 0 else 0
        # Output indices buffer (int64)
        out_idx = torch.empty(N, dtype=torch.int64, device=device)

        # Launch Triton sort kernel: grid axis=0 = N, axis=1 = LOGN
        bitonic_sort_stable_indices_1d[(N, LOGN)](flat, out_idx, N, LOGN=LOGN, num_warps=4)

        # 2) Compute expert_offsets using Triton histogram via atomic adds, then cumsum and cast
        num_experts = 256  # default in original
        offsets_int64 = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)

        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_histogram_atomic_kernel[grid](flat, offsets_int64, N, num_experts, BLOCK=BLOCK, num_warps=4)

        cumsum_int64 = offsets_int64[1:].cumsum(0)  # shape (num_experts,)
        expert_offsets = cumsum_int64.to(torch.int32)  # shape (num_experts + 1,), last element ignored

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
