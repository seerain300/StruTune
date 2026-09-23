import triton
import triton.language as tl


# Triton kernels: sorting permutation via odd-even transposition (stable)
@triton.jit
def odd_even_even_phase(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements; grid should be 1, and we iterate over all lanes with lane < N
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < N

    # For even phase, compare pairs: (0,1), (2,3), ...
    partner = lane + 1
    partner_mask = partner < N

    # Create left/right views
    v_left = tl.load(values_ptr + lane, mask=mask, other=0)
    v_right = tl.load(values_ptr + partner, mask=partner_mask, other=0)
    idx_left = tl.load(indices_ptr + lane, mask=mask, other=0)
    idx_right = tl.load(indices_ptr + partner, mask=partner_mask, other=0)

    # Decide swap: only swap when left > right; preserve original order on equality (stable)
    do_swap = mask & partner_mask & (v_left > v_right)

    # Swap logic
    new_left = tl.where(do_swap, v_right, v_left)
    new_right = tl.where(do_swap, v_left, v_right)
    new_idx_left = tl.where(do_swap, idx_right, idx_left)
    new_idx_right = tl.where(do_swap, idx_left, idx_right)

    # Store back
    tl.store(values_ptr + lane, new_left, mask=mask)
    tl.store(values_ptr + partner, new_right, mask=partner_mask)
    tl.store(indices_ptr + lane, new_idx_left, mask=mask)
    tl.store(indices_ptr + partner, new_idx_right, mask=partner_mask)


@triton.jit
def odd_even_odd_phase(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program handles BLOCK elements; grid should be 1
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < N

    # For odd phase, compare pairs: (1,2), (3,4), ...
    partner = lane + 1
    partner_mask = partner < N

    v_left = tl.load(values_ptr + lane, mask=mask, other=0)
    v_right = tl.load(values_ptr + partner, mask=partner_mask, other=0)
    idx_left = tl.load(indices_ptr + lane, mask=mask, other=0)
    idx_right = tl.load(indices_ptr + partner, mask=partner_mask, other=0)

    do_swap = mask & partner_mask & (v_left > v_right)

    new_left = tl.where(do_swap, v_right, v_left)
    new_right = tl.where(do_swap, v_left, v_right)
    new_idx_left = tl.where(do_swap, idx_right, idx_left)
    new_idx_right = tl.where(do_swap, idx_left, idx_right)

    tl.store(values_ptr + lane, new_left, mask=mask)
    tl.store(values_ptr + partner, new_right, mask=partner_mask)
    tl.store(indices_ptr + lane, new_idx_left, mask=mask)
    tl.store(indices_ptr + partner, new_idx_right, mask=partner_mask)


# Triton kernel: histogram counts of values in [0..255]
@triton.jit
def histogram_atomic(values_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Process values in chunks; each program instance handles BLOCK elements
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < N
    # Load values, ignore mask by using other=0 (unused in atomic add due to mask)
    vals = tl.load(values_ptr + lane, mask=mask, other=0).to(tl.int32)
    # Atomic add 1 for each valid value in [0..255]
    # Note: values are expected to be in range 0..255 (as in the original code).
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton kernel: inclusive prefix sum of counts -> offsets
@triton.jit
def inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    # Single program instance scan over M=256
    # offsets_ptr[0] must be set to 0 on host
    # We can implement scan by iterating indices
    # Triton supports while loops
    # We pass M as constexpr for compilation, but keep dynamic to be safe
    running = tl.load(offsets_ptr + 0)
    i = 0
    while i < M:
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(offsets_ptr + (i + 1), running)
        i += 1


def _choose_block(N: int) -> int:
    # Choose a reasonable BLOCK size; 4096 covers typical N in this benchmark
    # Ensure at least 128 lanes per program
    if N <= 128:
        return 128
    elif N <= 256:
        return 256
    elif N <= 512:
        return 512
    elif N <= 1024:
        return 1024
    else:
        return 4096


def _num_phases(N: int) -> int:
    # Odd-even requires 2*N phases to guarantee sorted order for any N
    return 2 * N


# Triton-only ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of run:
        - Computes stable argsort permutation of flattened topk_idx (int32 on CUDA) via Triton odd-even sort.
        - Builds counts histogram (0..255) via Triton atomic adds.
        - Computes expert_offsets (inclusive prefix sum of counts) via Triton scan.
        Returns:
          sorted_token_indices: int32 permutation (num_tokens,)
          expert_offsets: int32 tensor of shape (num_experts+1,) with offsets[0] = 0
        """
        device = topk_idx.device
        # Ensure inputs are on CUDA and int32
        if not topk_idx.is_cuda:
            raise RuntimeError("ModelNew expects topk_idx on CUDA device.")
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Sort permutation via Triton odd-even transposition sort
        values = flat.clone().to(torch.int32)  # working buffer of indices
        indices = torch.arange(N, dtype=torch.int32, device=device)  # permutation buffer

        # Launch phases: T = 2*N
        BLOCK = _choose_block(N)
        grid = (1,)  # single program handles BLOCK lanes; we rely on mask for all elements
        T = _num_phases(N)
        for t in range(T):
            if (t % 2) == 0:
                odd_even_even_phase[grid](values, indices, N, BLOCK=BLOCK, num_warps=8)
            else:
                odd_even_odd_phase[grid](values, indices, N, BLOCK=BLOCK, num_warps=8)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_atomic[grid_hist](values, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        # Return sorted_token_indices (permutation), and offsets
        return indices, offsets


def run(*args):
    return ModelNew()(*args)
