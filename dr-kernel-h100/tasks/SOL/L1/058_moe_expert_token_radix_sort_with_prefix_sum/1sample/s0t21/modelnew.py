import torch
import triton
import triton.language as tl


@triton.jit
def _argsort_odd_even_values_indices(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program instance handles a tile of BLOCK elements.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Initialize working values from input and indices buffer [0..N-1]
    v = tl.load(values_ptr + offsets, mask=mask, other=0)
    idx = offsets  # original positions
    # Pointers for values and indices
    v_ptr = values_ptr + offsets
    idx_ptr = indices_ptr + offsets

    # Odd-even transposition sort: N phases
    # We'll run a fixed number of iterations (e.g., 1024) which is more than enough for typical N.
    # Each iteration performs:
    # - Even phase: compare-swap pairs (0,1), (2,3), ...
    # - Odd phase:  compare-swap pairs (1,2), (3,4), ...
    # Stability: only swap if v_i > v_j; equal values keep original order (idx smaller comes first).

    # Note: We update both values and indices together for each pair.
    # Even phase
    for it in range(0, 1024):
        is_even = (it % 2) == 0
        # Determine partner for each lane
        # For even phase: partner = i+1 if i is even
        # For odd phase:  partner = i+1 if i is odd
        if is_even:
            partner = offsets + 1
        else:
            partner = offsets + 1  # this will be masked appropriately below

        # Only process each pair once and ensure valid partner
        if is_even:
            process = (offsets % 2 == 0) & (offsets + 1 < N)
        else:
            process = (offsets % 2 == 1) & (offsets + 1 < N)

        # Load partner values and indices
        v_partner = tl.load(values_ptr + partner, mask=process, other=0)
        idx_partner = partner  # positions

        # Stable compare-swap: swap if v > v_partner, else keep. For equal values, do not swap.
        swap = v > v_partner

        new_v_i = tl.where(swap, v_partner, v)
        new_v_j = tl.where(swap, v, v_partner)

        new_idx_i = tl.where(swap, idx_partner, idx)
        new_idx_j = tl.where(swap, idx, idx_partner)

        # Store back to positions i and j based on 'process'
        # For even phase, process masks out odd i; for odd phase, process masks out even i.
        # We need to store only once per pair; masked stores handle this.
        # Even phase: store for i (even offsets) and j (odd offsets) based on masks.
        # Odd phase:  store for i (odd offsets) and j (even offsets) based on masks.
        if is_even:
            # Only update even i and their partner odd j
            tl.store(v_ptr, new_v_i, mask=(process & (offsets % 2 == 0)))
            tl.store(idx_ptr, new_idx_i, mask=(process & (offsets % 2 == 0)))
            tl.store(v_ptr + 1, new_v_j, mask=(process & (offsets % 2 == 1)))
            tl.store(idx_ptr + 1, new_idx_j, mask=(process & (offsets % 2 == 1)))
        else:
            # Odd phase: only update odd i and their partner even j
            tl.store(v_ptr, new_v_i, mask=(process & (offsets % 2 == 1)))
            tl.store(idx_ptr, new_idx_i, mask=(process & (offsets % 2 == 1)))
            tl.store(v_ptr + 1, new_v_j, mask=(process & (offsets % 2 == 0)))
            tl.store(idx_ptr + 1, new_idx_j, mask=(process & (offsets % 2 == 0)))

    # After N phases, indices_ptr contains the permutation that sorts values ascending, stable.
    # We already wrote back to indices_ptr; nothing else to do here.


@triton.jit
def _histogram_atomic_kernel(vals_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(vals_ptr + offsets, mask=mask, other=0)  # int32
    # Atomic add into counts[vals] for valid lanes
    # counts_ptr is int32[256]; assume vals in [0, 255]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    # Single-program inclusive scan over M=256 counts into offsets_ptr[0..M]
    acc = 0
    for i in range(0, M):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Assume device is CUDA; inputs from get_inputs are on CUDA
        # Flatten and ensure int32
        flat = topk_idx.reshape(-1).to(torch.int32)

        N = flat.numel()

        # Allocate output permutation and working values
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        # We can use flat itself as working values buffer for the odd-even sort
        # but odd-even requires read-write pairs; using a separate buffer is safer.
        values = flat.clone()

        # Triton odd-even sort for stable permutation
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _argsort_odd_even_values_indices[grid](values, sorted_token_indices, N, BLOCK=BLOCK, num_warps=4)

        # Triton histogram
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid_hist](values, counts, N, BLOCK=BLOCK, num_warps=8)

        # Triton inclusive scan to build offsets
        offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets[0] = 0  # initialize first offset
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets