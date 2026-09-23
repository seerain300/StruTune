import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_argsort_stable(flat_ptr, out_idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Argsort the 1D array 'flat_ptr' of length N into 'out_idx_ptr' (indices 0..N-1).
    Bitonic sort network with stable tie-breaking by original index (smaller index first).
    We process all indices i in 0..BLOCK-1; for i >= N, we use masked loads to avoid OOB.
    """
    # Generate indices vector for lanes
    i = tl.arange(0, BLOCK)  # int32 vector
    # Original positions (0..N-1)
    orig = i  # positions of flat_ptr/out_idx_ptr

    # We will implement full bitonic network over BLOCK elements, masking out operations beyond N.
    # For each stage k and substage j, compare with partner = i ^ stride.
    # Only process each pair once (i < partner). We load/store using masks to avoid OOB and handle ties.
    for k in range(1, BLOCK):
        stride = 1 << (k - 1)
        for j in range(k, 0, -1):
            stride_sub = 1 << (j - 1)
            partner = i ^ stride_sub
            # Only process the lower index in each pair to avoid double-processing.
            pair_mask = partner > i  # since 'i' is a vector, this is elementwise but used with other masks

            # Validity masks
            valid_i = i < N
            valid_partner = partner < N
            do_pair = pair_mask & valid_i & valid_partner

            # Load values/indices
            vi = tl.load(flat_ptr + i, mask=valid_i, other=0)
            vj = tl.load(flat_ptr + partner, mask=valid_partner, other=0)
            ii = tl.load(out_idx_ptr + i, mask=valid_i, other=0)
            ij = tl.load(out_idx_ptr + partner, mask=valid_partner, other=0)

            # Ascending or descending direction for sequence i
            asc = ((i & k) == 0)  # sequence index, not N

            # Compare-exchange with stable tie-break by original index
            swap = (vi > vj) | ((vi == vj) & (ii > ij))  # ascending: swap if vi> vj or tie and ii > ij
            swap_desc = (vi < vj) | ((vi == vj) & (ii < ij))  # descending: swap if vi< vj or tie and ii < ij
            swap = tl.where(asc, swap, swap_desc) & do_pair

            new_vi = tl.where(swap, vj, vi)
            new_vj = tl.where(swap, vi, vj)
            new_ii = tl.where(swap, ij, ii)
            new_ij = tl.where(swap, ii, ij)

            # Store back to out_idx_ptr only for the lower index in each pair
            # For all i (not just lower), we can write final positions:
            # We write new_ii/new_ij for all valid i; non-pair elements do not swap, so new_* == original.
            # Implement by masked stores: only write for valid indices.
            tl.store(out_idx_ptr + i, new_ii, mask=valid_i)
            tl.store(out_idx_ptr + partner, new_ij, mask=valid_partner)


@triton.jit
def count_histogram_atomic(flat_ptr, N, counts_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Build histogram of values in flat_ptr[0..N) into counts_ptr[0..num_experts-1].
    Each program processes BLOCK elements and atomically adds 1 for each occurrence of the value.
    Assumes flat_ptr values are in [0, num_experts-1].
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # expect int32
    for o in range(BLOCK):
        idx = start + o
        if mask[idx]:
            val = vals[o]
            # Atomic add into counts[val]
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts into offsets_ptr[0..N_bins-1]:
    offsets[i] = sum_{k=0..i-1} counts[k] for i in 1..N_bins-1; offsets[0] = 0.
    This is O(N_bins^2), acceptable for N_bins=256.
    """
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version of the original Model:
    - sorted_token_indices: permutation of original flattened indices that sorts values ascending with stable=True.
      Implemented via Triton bitonic argsort (stable tie-break by original index).
    - expert_offsets: exclusive prefix sum over histogram of expert indices computed in Triton.
    No torch.sort, torch.bincount, or torch.cumsum in forward; Triton kernels are launched in forward.
    """
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton argsort (ascending, stable) into sorted_token_indices
        # We need length of output indices buffer. Create temporary out_idx initialized to 0..N-1.
        out_idx = torch.arange(N, dtype=torch.int32, device=device)

        # Choose BLOCK as next power-of-two >= N for bitonic network; here we set BLOCK=N if N is power of two.
        # For simplicity and correctness, set BLOCK=N and rely on mask for i >= N. Since N can be odd,
        # use next power of two.
        def next_pow2(x: int) -> int:
            return 1 << (x - 1).bit_length()
        BLOCK = next_pow2(N)

        # Launch bitonic argsort kernel
        bitonic_argsort_stable[(1,)](flat, out_idx, N, BLOCK, num_warps=4)

        # 2) Triton histogram (counts per expert) using parallel atomic adds
        counts = torch.zeros(256, dtype=torch.int32, device=device)

        grid = (triton.cdiv(N, 1024),)
        count_histogram_atomic[grid](flat, N, counts, num_experts=256, BLOCK=1024, num_warps=4)

        # 3) Triton exclusive prefix sum to produce expert_offsets: length = num_experts + 1
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=256, num_warps=1)

        # Return sorted_token_indices (int32) and expert_offsets (int32)
        # out_idx holds stable argsort indices (i.e., the positions of sorted values in flat).
        # sorted_token_indices = out_idx
        return out_idx, offsets


def run(*args):
    return ModelNew()(*args)
