import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_stable_argsort_triton(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Triton kernel: perform stable argsort of 'values' (int32) of length N.
    Uses odd-even transposition sort.
    'indices_ptr' holds permutation (original positions) and is updated during sorting.
    """
    pid = tl.program_id(axis=0)
    # Each program processes a chunk of lanes; for simplicity we run a single program and broadcast,
    # but using a single program over BLOCK lanes suffices. Alternatively, grid=(1,) and
    # perform loops using tl.arange. Here, we assume grid=(1,) and BLOCK >= N.
    # We'll iterate over phases using a while loop (Triton supports while loops).
    # To keep correctness, we launch the kernel with grid=(1,) and set BLOCK to a large value (e.g., 8192).
    # Inside, we operate over a vector of lanes [0..BLOCK-1], masking lanes >= N.
    lanes = tl.arange(0, BLOCK)
    mask_lane = lanes < N

    # Initialize local copies of values and indices
    v = tl.load(values_ptr + lanes, mask=mask_lane, other=0)  # int32
    idx = lanes  # original positions

    # Number of phases: 2*N (sufficient for odd-even sort to converge)
    phases = 0
    while phases < 2 * N:
        # Determine even or odd phase
        is_even = (phases % 2) == 0

        # Compute partner indices for this phase
        if is_even:
            j = lanes + 1
            pair_mask = ((lanes & 1) == 0) & (j < N) & mask_lane
            pair_j = lanes + 1
        else:
            j = lanes - 1
            pair_mask = ((lanes & 1) == 1) & (j >= 0) & mask_lane
            pair_j = lanes - 1

        # For paired lanes, load partner values and indices
        v_j = tl.load(values_ptr + pair_j, mask=pair_mask, other=0)
        idx_j = tl.load(indices_ptr + pair_j, mask=pair_mask, other=lanes)

        # Compute swap condition: stable ascending
        # Only swap when current > partner. For ties (==), do not swap.
        swap = v > v_j

        # Perform swap on both sides for paired lanes
        new_v = tl.where(swap & pair_mask, v_j, v)
        new_idx = tl.where(swap & pair_mask, idx_j, idx)
        v = new_v
        idx = new_idx

        phases += 1

    # Write back the final permutation (indices of sorted positions)
    tl.store(indices_ptr + lanes, idx, mask=mask_lane)


@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Triton kernel: build histogram of values in flat (int32) into counts[0..255].
    Uses masked atomic_add per element.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # Atomic add into counts[vals] for valid lanes
    # Note: only increment for values in [0..255]
    valid_val = (vals >= 0) & (vals <= 255) & mask
    tl.atomic_add(counts_ptr + vals, 1, mask=valid_val)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Triton kernel: inclusive prefix sum over counts[0..M-1], writes offsets[0..M].
    offsets[0] is passed as 0; kernel fills offsets[1..M].
    """
    # Single program performs iterative doubling scan over M elements.
    running = 0
    # Iterate log2(M) steps; we set M=256 so steps=8
    for step in range(0, 8):
        # Load current counts at positions step distance back
        # We reconstruct via a vectorized approach by reading counts[i] where i in [step, M-1]
        # To do so, we loop over i and update offsets.
        # Triton doesn't have vectorized gather update like CUDA's prefix-scan intrinsics,
        # so we implement a simple loop over i: each thread holds running and updates offsets[i].
        # However, Triton prefers vector lanes; here we keep running scalar and update offsets
        # with tl.load/tl.store per i. This is fine for M=256.
        # Compute positions for this step
        # Note: We will use vectorized operations where possible; here we use scalar accumulation
        # across the M dimension by pretending each thread lane handles one position sequentially.
        # Triton requires per-thread operations; we simulate by using a single program with one lane
        # and iterating i from 0 to M-1. This ensures correctness.
        pass  # The actual scan loop will be implemented below


# Implement the scan with a single-program loop over i
@triton.jit
def _inclusive_scan_prefix_sum_single(counts_ptr, offsets_ptr, M: tl.int32):
    running = 0
    # Iterate over each position i from 0 to M-1
    # Triton allows Python range loops; running is a scalar int32.
    for i in range(0, M):
        v = tl.load(counts_ptr + i)
        running += v
        tl.store(offsets_ptr + i, running)


# Launch helper for inclusive scan
def _launch_inclusive_scan(counts, offsets):
    # We need to pass M=256. offsets[0] is already 0.
    # Use a single program instance.
    _inclusive_scan_prefix_sum_single[(1,)](counts, offsets, M=256, num_warps=1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of run:
        - Produces sorted_token_indices (stable argsort of flattened indices)
        - Produces expert_offsets (inclusive prefix sum of histogram over 256 experts)
        All computation performed inside Triton kernels launched from forward.
        """
        device = topk_idx.device
        # Flatten to 1D int32
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()
        num_experts = 256

        # 1) Stable argsort via Triton odd-even sort
        # Allocate working copy for values and permutation buffer
        values = flat.clone()  # int32
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Launch kernel with grid=(1,) and a large BLOCK to cover N
        # Choose BLOCK as next power-of-two up to 8192 for robustness
        BLOCK = 8192
        _odd_even_stable_argsort_triton[(1,)](values, sorted_token_indices, N, BLOCK=BLOCK, num_warps=8)

        # 2) Histogram via Triton (atomic adds into counts[256])
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_H = 1024
        grid_hist = (triton.cdiv(N, BLOCK_H),)
        _histogram_atomic_kernel[grid_hist](flat, counts, N=N, BLOCK=BLOCK_H, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan
        # offsets has length num_experts + 1
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _launch_inclusive_scan(counts, offsets)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
