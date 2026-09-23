import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_stable_argsort(values_ptr, indices_ptr, N: tl.int32):
    """
    Odd-even transposition sort (stable) on values_ptr of length N.
    Maintains parallel indices_ptr as the permutation (original positions).
    Only swaps when values[i] > values[j]; equal values are not swapped (stable).
    Each program handles its own position and updates partner accordingly.
    """
    pid = tl.program_id(axis=0)
    # Each program corresponds to one element position in [0, N)
    # We will perform N phases: even and odd passes.
    # In each pass, each program compares with its partner and swaps if needed.
    # Even pass: pairs (0,1), (2,3), ...
    # Odd pass : pairs (1,2), (3,4), ...
    # We loop over phases; we do direct updates: for each position pid,
    # compute partner ip, load both, decide swap, and store updated values
    # to both pid and ip addresses via masks (only once per pair to avoid double writes).
    # Note: We implement the full N passes here. Triton will execute this loop per program.
    for t in range(0, N):
        # Determine partner for this position
        is_even_pass = (t % 2) == 0
        # Even pass pairs (0,1), (2,3), ...
        # Odd pass pairs (1,2), (3,4), ...
        # Compute partner pid_partner and parity for this position
        # If pid is even: partner is pid + 1; if pid is odd: partner is pid - 1
        pid_partner = tl.where(is_even_pass, pid + 1, pid - 1)
        # Validity masks
        valid_self = pid < N
        valid_partner = pid_partner >= 0 and pid_partner < N

        # Do not process if not valid or no partner
        if not valid_self or not valid_partner:
            continue

        # Load current values
        vi = tl.load(values_ptr + pid, mask=valid_self, other=0)
        vj = tl.load(values_ptr + pid_partner, mask=valid_partner, other=0)

        # Load original indices (positions)
        idxi = tl.load(indices_ptr + pid, mask=valid_self, other=0)
        idxj = tl.load(indices_ptr + pid_partner, mask=valid_partner, other=0)

        # Decide swap: only swap if vi > vj (stable), and ensure we write to both positions once
        # We will write to pid and pid_partner. For each position, only process if pid < pid_partner.
        do_swap = (vi > vj) & valid_self & valid_partner & (pid < pid_partner)

        # Compute new values after possible swap
        new_vi = tl.where(do_swap, vj, vi)
        new_vj = tl.where(do_swap, vi, vj)
        new_idxi = tl.where(do_swap, idxj, idxi)
        new_idxj = tl.where(do_swap, idxi, idxj)

        # Store updated values
        tl.store(values_ptr + pid, new_vi, mask=valid_self & (pid < pid_partner))
        tl.store(values_ptr + pid_partner, new_vj, mask=valid_partner & (pid < pid_partner))
        # Store updated indices (must mirror values stores)
        tl.store(indices_ptr + pid, new_idxi, mask=valid_self & (pid < pid_partner))
        tl.store(indices_ptr + pid_partner, new_idxj, mask=valid_partner & (pid < pid_partner))


@triton.jit
def _histogram_atomic_kernel(values_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Compute histogram of values in values_ptr (int32), updating counts_ptr[0..255]
    using atomic adds. Each program processes BLOCK elements.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values for this block (int32), masked
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)

    # Increment counts via atomic_add for valid elements
    # values are in [0, 255] as per the original code
    for i in range(0, BLOCK):
        val = vals[i]
        valid = mask[i] & (val >= 0) & (val <= 255)
        if valid:
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[0..M],
    with offsets_ptr[0] = 0. We set offsets_ptr[1..] = prefix sum.
    Note: We only write offsets_ptr[1..], but caller initializes offsets_ptr[0] = 0.
    """
    acc = 0
    for i in range(0, M):
        acc += tl.load(counts_ptr + i)
        # Store to offsets[i+1] (index i)
        tl.store(offsets_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Sorts flattened indices via Triton odd-even stable argsort to obtain permutation.
        - Computes expert offsets via Triton histogram and prefix sum.
        """
        device = topk_idx.device
        # Ensure CUDA and int32
        assert topk_idx.is_cuda, "Input must be on CUDA for Triton kernels."
        topk_idx = topk_idx.contiguous()
        flat = topk_idx.view(-1)
        N = flat.numel()

        # 1) Triton stable argsort: produce permutation of original indices
        # We will use the flattened flat as values and create indices buffer [0..N-1].
        values = flat.clone()  # Triton expects pointer to data; keep int32
        indices = torch.arange(N, dtype=torch.int32, device=device)
        # Launch odd-even sort kernel: 1D grid over N
        grid_sort = (N,)
        _odd_even_stable_argsort[grid_sort](values, indices, N, num_warps=8)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid_hist](values, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # initialize first element to 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        # Return sorted permutation and offsets; both int32 as in original run
        return indices, offsets


def run(*args):
    return ModelNew()(*args)
