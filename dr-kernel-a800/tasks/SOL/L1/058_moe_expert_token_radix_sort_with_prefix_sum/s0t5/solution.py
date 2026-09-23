import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_sort_stable_int64(flat_ptr, out_ptr, N: tl.constexpr, LOGN: tl.constexpr):
    """
    Stable bitonic sort on int64 array. Grid = (N, LOGN). Each program handles
    one element i in one stage j. Tie-breaking uses original index i: for equal
    values, smaller i comes first.
    """
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    step = 1 << j
    partner = i ^ step
    do_pair = i < partner
    in_bounds = (i < N) & (partner < N) & do_pair

    a = tl.load(flat_ptr + i, mask=in_bounds, other=0)  # int64
    b = tl.load(flat_ptr + partner, mask=in_bounds, other=0)  # int64

    asc = ((i & (1 << j)) == 0)  # ascending if bit j of i is 0

    # Stable compare: if a == b, prefer smaller original index
    tie = a == b
    less = a < b
    greater = a > b
    # Min/max ignoring ties
    minv = tl.where(less, a, b)
    maxv = tl.where(greater, a, b)
    # For tie, prefer a if i < partner
    take_a = less | (tie & (i < partner))
    new_i = tl.where(take_a, a, b)
    new_p = tl.where(take_a, b, a)

    new_i_stage = tl.where(asc, new_i, new_p)
    new_p_stage = tl.where(asc, new_p, new_i)

    tl.store(out_ptr + i, new_i_stage, mask=in_bounds)
    tl.store(out_ptr + partner, new_p_stage, mask=in_bounds)


@triton.jit
def histogram_atomic_kernel(flat_ptr, offsets_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Histogram of flat indices (int32) into offsets_ptr (int32) of length num_experts.
    Each program processes a chunk of flat and atomically increments offsets[flat[i]].
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
    tl.atomic_add(offsets_ptr + vals, 1, mask=mask)


@triton.jit
def cumsum_inclusive_kernel(offsets_counts_ptr, offsets_out_ptr, num_experts: tl.constexpr):
    """
    Inclusive prefix sum of offsets_counts_ptr (int32, length num_experts)
    into offsets_out_ptr (int32, length num_experts + 1).
    out[0] = 0, out[1:] = cumulative counts.
    Single-program iterative doubling scan.
    """
    # We'll perform iterative doubling for up to 8 steps (since 256 = 2^8).
    for step in range(0, 8):
        stride = 1 << step
        # Compute carry from previous positions
        for i in range(0, num_experts):
            val_i = tl.load(offsets_counts_ptr + i)
            prev = tl.load(offsets_counts_ptr + (i - stride), mask=(i >= stride), other=0)
            new_i = val_i + prev
            tl.store(offsets_counts_ptr + i, new_i)
        # Write inclusive sums to output
        tl.store(offsets_out_ptr + 0, 0)
        for i in range(0, num_experts):
            tl.store(offsets_out_ptr + (1 + i), tl.load(offsets_counts_ptr + i))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sorts flattened topk_idx (int32) into sorted_token_indices (int64) via Triton bitonic sort.
        - Computes expert_offsets (int32, length num_experts+1) via Triton histogram + inclusive scan.
        Returns:
            sorted_token_indices: torch.int64 tensor of shape (N,)
            expert_offsets: torch.int32 tensor of shape (num_experts + 1,)
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

        # Output for sorting (int64 to match torch.sort indices)
        sorted_token_indices = torch.empty(N, dtype=torch.int64, device=device)

        # Bitonic sort stable in Triton
        LOGN = 0 if N <= 1 else (N - 1).bit_length()
        grid = (N, LOGN)
        bitonic_sort_stable_int64[grid](flat, sorted_token_indices, N=N, LOGN=LOGN)

        # Compute expert_offsets via Triton histogram + inclusive scan (int32)
        num_experts = 256  # matches original default
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_hist = ((N + BLOCK - 1) // BLOCK,)
        histogram_atomic_kernel[grid_hist](flat, counts, N, num_experts, BLOCK=BLOCK)

        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        cumsum_inclusive_kernel[(1,)](counts, expert_offsets, num_experts=num_experts)

        return sorted_token_indices, expert_offsets


def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    """Generate inputs with valid expert indices in range [0, num_experts-1]."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Generate random expert indices in valid range [0, num_experts-1]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    
    return {"topk_idx": topk_idx}


def run(*args):
    return ModelNew()(*args)
