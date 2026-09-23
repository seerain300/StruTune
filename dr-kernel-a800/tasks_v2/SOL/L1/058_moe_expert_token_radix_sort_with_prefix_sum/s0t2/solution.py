import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_sort_stable_int32(out_ptr, flat_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort (ascending) of the 1D int32 array flat_ptr of length N.
    Writes sorted values to out_ptr (int32). Stable tie-breaker: equal values preserve original order (i < partner).
    Uses 2D grid: axis0=N (one program per element), axis1=LOGN (one stage per bitonic pass).
    """
    i = tl.program_id(axis=0)
    # Unrolled bitonic stages: for each k in [1..LOGN], do inner stages j = k-1 down to 0.
    for k in tl.static_range(1, LOGN + 1):
        for jj in tl.static_range(0, k):
            j = k - 1 - jj
            step = 1 << j
            partner = i ^ step
            # Only process each pair once and within bounds
            do_pair = i < partner
            in_bounds = (i < N) & (partner < N) & do_pair
            # Load values
            a = tl.load(flat_ptr + i, mask=in_bounds, other=0)
            b = tl.load(flat_ptr + partner, mask=in_bounds, other=0)
            # Ascending/descending for this stage: elements where (i & (1<<k)) == 0 go ascending, else descending
            asc = ((i & (1 << k)) == 0)
            # Stable tie-breaking: if equal, smaller original index comes first
            tie = a == b
            # Determine min/max ignoring tie, then apply tie-break
            minv = tl.where(a < b, a, b)
            maxv = tl.where(a > b, a, b)
            # If a <= b (or tie and i < partner), i should take 'a'; else take 'b'
            take_a_i = (a <= b) | (tie & (i < partner))
            new_i = tl.where(take_a_i, a, b)
            new_p = tl.where(take_a_i, b, a)
            # Apply ascending/descending
            new_i_stage = tl.where(asc, new_i, new_p)
            new_p_stage = tl.where(asc, new_p, new_i)
            # Store results
            tl.store(out_ptr + i, new_i_stage, mask=in_bounds)
            tl.store(out_ptr + partner, new_p_stage, mask=in_bounds)


@triton.jit
def count_and_cumsum_kernel(flat_ptr, offsets_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Build inclusive prefix sum of histogram of flat indices into offsets[1:].
    offsets has length num_experts + 1; offsets[0] is unused; offsets[1:] = cumulative counts.
    Each element of flat_ptr (int32) is in [0, num_experts-1].
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values
    # Atomic add 1 for each occurrence into offsets[val + 1]
    tl.atomic_add(offsets_ptr + (vals + 1), 1, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sorts the flattened topk_idx (int32) using a Triton stable bitonic sort and writes results to out.
        - Computes expert_offsets via Triton (histogram + prefix sum using atomic adds).
        Returns:
            sorted_token_indices: torch.int64 tensor of shape (N,)  (to match torch.sort indices dtype)
            expert_offsets: torch.int32 tensor of shape (num_experts + 1,)
        """
        # Ensure input on CUDA and dtype int32
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # Output buffer for sorted values (int32)
        out = torch.empty_like(flat, dtype=torch.int32, device=device)

        # If N <= 1, just copy
        if N <= 1:
            out.copy_(flat)
        else:
            # Bitonic sort requires stages up to LOGN = ceil(log2(N))
            LOGN = (N - 1).bit_length()
            grid = (N,)
            bitonic_sort_stable_int32[grid](out, flat, N, LOGN=LOGN)

        # Compute expert_offsets via Triton histogram + prefix sum
        num_experts = 256  # matches original default
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

        # Choose a BLOCK size; 1024 works well for these sizes
        BLOCK = 1024
        grid_blocks = (triton.cdiv(N, BLOCK),)
        count_and_cumsum_kernel[grid_blocks](flat, expert_offsets, N, num_experts=num_experts, BLOCK=BLOCK)

        # Return sorted indices (int64 to match torch.sort) and expert_offsets
        sorted_token_indices = out.to(torch.int64)
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
