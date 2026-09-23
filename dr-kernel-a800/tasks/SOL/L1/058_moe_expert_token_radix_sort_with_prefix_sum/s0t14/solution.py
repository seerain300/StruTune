import torch
import triton
import triton.language as tl


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
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    pid = tl.program_id(axis=0)
    # Single program computes the prefix sum sequentially
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). out_idx_ptr holds int64 original positions 0..N-1.
    For each bitonic stage (k, j), each program i compares with partner = i ^ (1 << j).
    Ascending if (i & (1 << (k+1))) == 0 else descending. Stability: tie-break by i < partner.
    """
    # We use a 2D grid: axis 0 = N, axis 1 = LOGN (sub-stages). Triton requires grid be a tuple.
    # Each program handles one element i (axis=0). The second axis is used to iterate stages
    # via kernel launch. Here we implement per-stage logic using compile-time loops, so
    # we pass LOGN as a constexpr and unroll stages.

    # Note: Triton doesn't allow a 2D grid to be used as loop axes directly. Instead, we
    # structure the kernel so that each program i performs all necessary stages using masks.
    # However, Triton requires the kernel body to have only one tl.program_id usage. To work around,
    # we implement the bitonic network using nested loops over k and j, where j runs from k+1 to LOGN-1.
    # This way, each program i runs the same sequence of pairwise operations for all stages.

    # Each program i operates on its own position; we initialize out_idx_ptr with identity.
    # Then, we perform bitonic compare-and-swap stages.

    # We need to create a temporary partner vector and perform masked pairwise swaps.
    # Triton allows broadcasting of tl.arange and bitwise XOR, so we can compute partner for each k, j.

    # To do this correctly, we restructure: use axis 0 to identify i, and keep the loop logic in-kernel.
    # We will rely on Triton unrolling since LOGN is constexpr.

    # Since Triton requires a single tl.program_id(axis=0), we will:
    # - Initialize out_idx_ptr to identity int64 (0..N-1).
    # - Then perform all stages. But Triton does not support direct out initialization per program.
    # Therefore, we pass pre-filled out_idx_ptr as input indices (identity), and sort in-place.

    # We cannot truly rely on pre-filling; hence we will:
    # - Initialize out_idx_ptr on host to torch.arange(N, device=..., dtype=torch.int64).
    # - Then run the kernel to sort it in-place. This is acceptable and matches torch.sort output.

    # However, Triton kernels operate on device memory; we must ensure out_idx_ptr is pre-filled.
    # The host will do that: forward creates out_idx = torch.arange(N, device=..., dtype=torch.int64).
    # Now, we implement the sorting network.

    # Triton doesn't support dynamic loop variables with Python range; but since LOGN is constexpr,
    # we can use Python for-loops over k and j. Triton will unroll them.

    # For each stage: k in 0..LOGN-1
    for k in range(0, LOGN):
        # j runs from k+1 to LOGN-1 (sub-stages within dimension 2^j)
        # We need to know LOGN (passed as constexpr). We can only use j up to LOGN-1.
        # Triton supports nested loops with Python int. We implement:
        for j in range(k + 1, LOGN):
            # partner = i ^ (1 << j)
            step = 1 << j
            partner = i ^ step
            # Bounds and pair condition
            in_bounds = (i < N) & (partner < N)
            # Load current indices and values at those positions
            idx_i = tl.load(out_idx_ptr + i)  # int64 index
            idx_p = tl.load(out_idx_ptr + partner)  # int64 index of partner
            val_i = tl.load(flat_ptr + idx_i)  # int32 value
            val_p = tl.load(flat_ptr + idx_p)  # int32 value of partner

            # Ascending/descending for this stage determined by k+1 bit
            asc = ((i & (1 << (k + 1))) == 0)

            # Stable tie-breaking: if equal, i < partner should come first
            tie = val_i == val_p
            less = val_i < val_p
            greater = val_i > val_p

            # Determine min/max ignoring tie, then apply tie-break
            minv = tl.where(less, val_i, val_p)
            maxv = tl.where(greater, val_i, val_p)
            # If val_i <= val_p (or tie and i < partner), i should take val_i; else take val_p
            take_i = (less | (tie & (i < partner))).to(tl.int1)
            new_i = tl.where(take_i, val_i, val_p)
            new_p = tl.where(take_i, val_p, val_i)

            # Apply ascending/descending
            new_i = tl.where(asc, new_i, new_p)
            new_p = tl.where(asc, new_p, new_i)

            # Store back only for valid pairs
            # Write new indices for i and partner
            # Note: We only write for i < partner to avoid double writes.
            tl.store(out_idx_ptr + i, new_i, mask=in_bounds & (i < partner))
            tl.store(out_idx_ptr + partner, new_p, mask=in_bounds & (i < partner))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sorts the flattened topk_idx (int32) using a Triton stable bitonic sort, returning int64 indices.
        - Computes expert_offsets via Triton (histogram + prefix sum using atomic adds + sequential prefix).
        Returns:
            sorted_token_indices: torch.int64 tensor of shape (N,)
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

        # Compute LOGN for bitonic sort
        LOGN = flat.numel().bit_length() if flat.numel() > 0 else 0

        # Output buffer for indices (initialize to identity int64)
        out_idx = torch.arange(N, device=device, dtype=torch.int64)

        # Launch stable bitonic sort: one program per element, stages unrolled
        stable_bitonic_sort_kernel[(N,)](flat, out_idx, N, LOGN=LOGN)

        # Compute expert_offsets via Triton histogram + prefix sum (int64)
        num_experts = 256
        counts = torch.zeros(num_experts, device=device, dtype=torch.int32)
        # Count histogram (int32 counts)
        grid_counts = (triton.cdiv(N, 1024),)
        count_histogram_kernel[grid_counts](flat, counts, N, num_experts=num_experts)

        # Compute inclusive prefix sum of counts (int64 offsets)
        offsets_int64 = torch.empty(num_experts + 1, device=device, dtype=torch.int64)
        offsets_int64[0] = 0
        grid_ps = (1,)
        prefix_sum_kernel[grid_ps](counts, offsets_int64, num_experts=num_experts)

        # Cast to int32 to match original expert_offsets
        expert_offsets = offsets_int64.to(torch.int32)[1:]

        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
