import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, num_experts: tl.int32):
    pid = tl.program_id(0)
    # Each program handles a chunk; atomic adds aggregated across all pids.
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values
    # Atomic add 1 for each valid element to its bin. Bounds checked by num_experts.
    # counts_ptr is int32[256]
    for v in vals:
        if mask[v]:
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def inclusive_scan_inplace(counts_ptr, BLOCK: tl.constexpr, LOG: tl.constexpr):
    # In-kernel inclusive scan for an int32 array of length BLOCK (here BLOCK=256).
    # We operate over the single array with grid=1, doing Hillis–Steele in LOG passes.
    idx = tl.program_id(0) * 128 + tl.arange(0, 128)
    mask = idx < BLOCK
    # We need to implement iterative doubling; Triton doesn't support arbitrary dynamic loops,
    # but we can perform a fixed number of passes (LOG=8). We'll do it manually for BLOCK=256.
    # Pass 1 (stride 1)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 1) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 1) == 1), tl.load(counts_ptr + (idx ^ 1), mask=mask & ((idx & 1) == 1), other=0), 0), mask=mask)
    # Pass 2 (stride 2)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 2) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 2) == 2), tl.load(counts_ptr + (idx ^ 2), mask=mask & ((idx & 2) == 2), other=0), 0), mask=mask)
    # Pass 3 (stride 4)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 4) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 4) == 4), tl.load(counts_ptr + (idx ^ 4), mask=mask & ((idx & 4) == 4), other=0), 0), mask=mask)
    # Pass 4 (stride 8)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 8) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 8) == 8), tl.load(counts_ptr + (idx ^ 8), mask=mask & ((idx & 8) == 8), other=0), 0), mask=mask)
    # Pass 5 (stride 16)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 16) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 16) == 16), tl.load(counts_ptr + (idx ^ 16), mask=mask & ((idx & 16) == 16), other=0), 0), mask=mask)
    # Pass 6 (stride 32)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 32) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 32) == 32), tl.load(counts_ptr + (idx ^ 32), mask=mask & ((idx & 32) == 32), other=0), 0), mask=mask)
    # Pass 7 (stride 64)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 64) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 64) == 64), tl.load(counts_ptr + (idx ^ 64), mask=mask & ((idx & 64) == 64), other=0), 0), mask=mask)
    # Pass 8 (stride 128)
    tmp = tl.load(counts_ptr + idx, mask=mask, other=0)
    contrib = tl.where(mask & ((idx & 128) == 0), tmp, 0)
    tl.store(counts_ptr + idx, tmp + tl.where(mask & ((idx & 128) == 128), tl.load(counts_ptr + (idx ^ 128), mask=mask & ((idx & 128) == 128), other=0), 0), mask=mask)


@triton.jit
def stable_bitonic_sort_inplace(vals_ptr, idx_ptr, N: tl.int32, BLOCK: tl.constexpr, LOG: tl.constexpr):
    # In-kernel bitonic argsort over BLOCK lanes, first N initialized from input, rest padded to MAX_INT.
    # We implement bitonic sorting network for indices, using tie-break by original index for stability.
    for stage in range(LOG):
        k = 1 << (LOG - stage - 1)
        for j in range(stage, -1, -1):
            stride = 1 << j
            partner = tl.arange(0, BLOCK) ^ stride
            # Only process each pair once: i < partner
            active = tl.arange(0, BLOCK) < partner
            v_i = tl.load(vals_ptr + tl.arange(0, BLOCK), mask=active, other=0)
            v_p = tl.load(vals_ptr + partner, mask=active, other=0)
            i_i = tl.load(idx_ptr + tl.arange(0, BLOCK), mask=active, other=0)
            i_p = tl.load(idx_ptr + partner, mask=active, other=0)
            # Compare-exchange with stable tie-break by original index
            swap = (v_i > v_p) | ((v_i == v_p) & (i_i > i_p))
            new_v_i = tl.where(swap, v_p, v_i)
            new_v_p = tl.where(swap, v_i, v_p)
            new_i_i = tl.where(swap, i_p, i_i)
            new_i_p = tl.where(swap, i_i, i_p)
            # Store back
            tl.store(vals_ptr + tl.arange(0, BLOCK), new_v_i, mask=active)
            tl.store(vals_ptr + partner, new_v_p, mask=active)
            tl.store(idx_ptr + tl.arange(0, BLOCK), new_i_i, mask=active)
            tl.store(idx_ptr + partner, new_i_p, mask=active)


def _next_power_of_two(n: int) -> int:
    # Minimum power-of-two >= n
    return 1 << (n - 1).bit_length()


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten (metadata-only)
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device

        N = flat.numel()
        num_experts = 256

        # 1) Histogram in Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Choose grid size for histogram; each program handles 128 elements
        BLOCK_HIST = 128
        grid_hist = (N + BLOCK_HIST - 1) // BLOCK_HIST
        histogram_kernel[(grid_hist,)](flat, counts, N, num_experts)

        # 2) Inclusive scan of counts in Triton (256 elements). Use a single-program pass with fixed steps.
        # Build a temporary counts_scan tensor of length num_experts
        counts_scan = counts  # in-place scan
        # We need grid=1 and LOG=8 passes (since 256 = 2**8)
        inclusive_scan_inplace[(1,)](counts_scan, BLOCK=256, LOG=8)

        # Build expert_offsets: [0, sum(0), sum(0..1), ..., sum(all)]
        # The last element equals N (sum of all counts)
        # We already have inclusive prefix in counts_scan
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[1:] = counts_scan

        # 3) Triton bitonic argsort to produce sorted_token_indices
        # Choose BLOCK_SORT = next power-of-two >= N, capped at 4096
        BLOCK_SORT = min(4096, _next_power_of_two(N))
        # LOG_SORT = number of stages = log2(BLOCK_SORT)
        LOG_SORT = int(BLOCK_SORT.bit_length() - 1)
        MAX_INT = (1 << 31) - 1

        vals = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)
        idx_out = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)

        # Initialize vals and idx_out
        vals[:N] = flat
        vals[N:] = MAX_INT
        idx_out[:N] = torch.arange(N, device=device)

        # Launch bitonic sort (argsort) in-place on idx_out
        stable_bitonic_sort_inplace[(1,)](vals, idx_out, N, BLOCK=BLOCK_SORT, LOG=LOG_SORT)

        # sorted_token_indices: first N entries of idx_out
        sorted_token_indices = idx_out[:N]

        # Return as int32 per original
        return sorted_token_indices.to(torch.int32), expert_offsets