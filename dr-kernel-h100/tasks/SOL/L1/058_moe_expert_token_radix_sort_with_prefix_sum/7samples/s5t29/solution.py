import torch
import triton
import triton.language as tl


@triton.jit
def stable_sort_bitonic_pairs(in_ids_ptr, out_ids_ptr, out_idx_ptr, N: tl.int32, LOGN: tl.constexpr, STEPS: tl.constexpr):
    """
    Bitonic sort on pairs (ID, index) to produce stable ascending order by ID.
    We operate over a power-of-two length P = 1 << LOGN, padding unused positions with large sentinel ID
    so they sort to the end. Only the first N elements are considered.
    """
    P = 1 << LOGN
    # For each stage in the bitonic network
    for s in tl.static_range(1, STEPS + 1):
        k = 1 << s
        for j in tl.static_range(s - 1, -1, -1):
            i = 1 << j
            partner = tl.arange(0, P) ^ k
            # Only process each pair once: take min(i, partner)
            mask_partner = partner < tl.arange(0, P)
            # Load current and partner values
            a_ids = tl.load(in_ids_ptr + tl.arange(0, P))
            a_idx = tl.load(out_idx_ptr + tl.arange(0, P))
            b_ids = tl.load(in_ids_ptr + partner, mask=mask_partner, other=0)  # other=0 used only if mask is False
            b_idx = tl.load(out_idx_ptr + partner, mask=mask_partner, other=0)

            # Stable compare: sort by ID ascending, tie-breaker by original index (smaller index first)
            # We'll update only indices where i < partner to avoid double writes.
            update_mask = tl.arange(0, P) < partner

            # Compute compare results and choose swapped or not
            # compare = (a_ids <= b_ids) or ((a_ids == b_ids) and (a_idx >= b_idx))
            compare = (a_ids <= b_ids) | ((a_ids == b_ids) & (a_idx >= b_idx))
            swap = (~update_mask) & compare

            new_a_ids = tl.where(swap, b_ids, a_ids)
            new_a_idx = tl.where(swap, b_idx, a_idx)
            new_b_ids = tl.where(swap, a_ids, b_ids)
            new_b_idx = tl.where(swap, a_idx, b_idx)

            # Store back
            tl.store(out_ids_ptr + tl.arange(0, P), new_a_ids, mask=update_mask)
            tl.store(out_idx_ptr + tl.arange(0, P), new_a_idx, mask=update_mask)
            tl.store(out_ids_ptr + partner, new_b_ids, mask=mask_partner & update_mask)
            tl.store(out_idx_ptr + partner, new_b_idx, mask=mask_partner & update_mask)


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Histogram of expert IDs in flat using chunked iteration and atomic_add.
    Assumes flat is 1D contiguous tensor of int32 values in [0, num_experts-1].
    """
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < N
        ids = tl.load(flat_ptr + idx, mask=mask, other=0)
        # For valid elements, atomic add 1 to counts[ids]
        for b in tl.static_range(0, BLOCK):
            id_val = ids[b]
            valid = mask[b]
            if valid:
                tl.atomic_add(counts_ptr + id_val, 1)
        offset += BLOCK


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute offsets[j] = sum_{i<j} counts[i] for j in 1..num_experts.
    offsets[0] is set on host.
    """
    # We assume offsets_ptr length >= num_experts + 1 and offsets[0] = 0 on host.
    total = tl.zeros((), dtype=tl.int32)
    for j in tl.static_range(1, num_experts + 1):
        prev = total
        total += tl.load(counts_ptr + (j - 1))
        tl.store(offsets_ptr + j, prev)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sort flattened expert IDs stably and return permutation indices.
        - Compute expert offsets via Triton histogram + exclusive prefix sum.
        """
        device = topk_idx.device
        num_experts = 256  # as in original setup

        # Flatten and ensure int32
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()

        # Bitonic sort requires power-of-two length; pad to next power of two
        # Compute LOGN = ceil(log2(max(1, N)))
        LOGN = 0
        if N <= 1:
            LOGN = 1
        else:
            LOGN = (N - 1).bit_length()
        P = 1 << LOGN
        STEPS = LOGN  # number of stages

        # Allocate outputs for sort
        out_ids = torch.empty(P, dtype=torch.int32, device=device)
        out_idx = torch.empty(P, dtype=torch.int32, device=device)

        # Initialize: out_idx = 0..P-1, out_ids = flat + padding (pad with sentinel -1)
        out_idx[:] = torch.arange(P, dtype=torch.int32, device=device)
        out_ids[:N] = flat
        # Pad positions with sentinel ID (must sort to end). -1 ensures ascending sort pushes them to the end.
        out_ids[N:] = -1

        # Launch bitonic sort kernel
        stable_sort_bitonic_pairs(out_ids, out_idx, N, LOGN=LOGN, STEPS=STEPS)

        # Extract sorted permutation indices (first N)
        sorted_token_indices = out_idx[:N].to(torch.int32)

        # Count expert IDs using Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Compute offsets using Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
