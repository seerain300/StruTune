import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, num_experts: tl.int32):
    pid = tl.program_id(0)
    # Each program handles one element
    if pid < N:
        val = tl.load(flat_ptr + pid)
        # Atomic add into counts[val]
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_scan_inplace(counts_ptr, out_ptr, E: tl.constexpr, LOG: tl.constexpr):
    # Hillis–Steele inclusive scan on a fixed-size array (E=256)
    # We perform LOG passes with vectorized lane updates:
    # out[i] += out[i - stride] for stride in 1, 2, 4, ..., 128
    idx = tl.arange(0, E)
    # Initialize out with counts
    out = tl.load(counts_ptr + idx)
    for t in range(LOG):
        stride = 1 << t
        # For lanes i >= stride, add out[i - stride]
        addend = tl.where(idx >= stride, out[idx - stride], 0)
        out += addend
        # Write back to out_ptr (same storage)
        tl.store(out_ptr + idx, out)


@triton.jit
def stable_bitonic_argsort(vals_ptr, idx_out_ptr, N: tl.int32, BLOCK: tl.constexpr, LOG: tl.constexpr):
    # Stable bitonic sort network that returns indices. We avoid sentinel padding by using next power-of-two BLOCK.
    lanes = tl.arange(0, BLOCK)
    # Initialize idx_out with original positions 0..BLOCK-1
    tl.store(idx_out_ptr + lanes, lanes)

    # Bitonic sorting network: i in 0..BLOCK-1
    i = lanes
    # Precompute all k for the network (compile-time constant LOG)
    for k in range(1, LOG + 1):
        j = 1 << (k - 1)
        for q in range(k - 1, -1, -1):
            ixj = 1 << q  # inner loop over j/2, j/4, ..., 1
            partner = i ^ ixj
            # Guard: only process each pair once
            take = partner > i
            # Load current vals for i and partner
            val_i = tl.load(vals_ptr + i)
            val_p = tl.load(vals_ptr + partner)
            idx_i = tl.load(idx_out_ptr + i)
            idx_p = tl.load(idx_out_ptr + partner)
            # Determine ascending direction for this subsequence
            ascending = (i & j) == 0
            # Stable compare: if values equal, use original index for tie-break
            tie = val_i == val_p
            cmp_val = (val_i < val_p) | (tie & (idx_i < idx_p))
            # If not ascending and should swap, or ascending and should not swap
            swap = (not ascending) & cmp_val | ascending & (not cmp_val)
            # Perform swap
            new_i = partner
            new_p = i
            new_idx_i = idx_p
            new_idx_p = idx_i
            # Write back for lanes where take is True
            tl.store(idx_out_ptr + i, tl.where(swap, new_idx_p, tl.load(idx_out_ptr + i)), mask=take)
            tl.store(idx_out_ptr + partner, tl.where(swap, new_idx_i, tl.load(idx_out_ptr + partner)), mask=take)
            # Also update vals consistently
            tl.store(vals_ptr + i, tl.where(swap, val_p, tl.load(vals_ptr + i)), mask=take)
            tl.store(vals_ptr + partner, tl.where(swap, val_i, tl.load(vals_ptr + partner)), mask=take)


def _next_power_of_two(n: int) -> int:
    # Returns next power of two >= n
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def _log2(n: int) -> int:
    # Returns number of passes for power-of-two n
    return n.bit_length() - 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten to 1D (N elements)
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton histogram: counts per expert
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # Launch one program per element
        grid = (N,)
        histogram_kernel[grid](flat, counts, N, 256)

        # 2) Triton inclusive scan to compute expert offsets (length 257)
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        # Inclusive scan on counts[0..255]
        inclusive_scan_inplace[(1,)](counts, offsets[1:], 256, 8)
        # Ensure offsets[256] = N (final cumulative count)
        if N not in (offsets[256],):
            offsets[256] = N

        # 3) Triton bitonic argsort (stable tie-breaking by original index)
        # Use next power-of-two block size
        BLOCK_SORT = _next_power_of_two(N)
        LOG_SORT = _log2(BLOCK_SORT)
        # We need scratch vals and idx_out of size BLOCK_SORT; we will read first N only
        vals = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)
        idx_out = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)
        # Initialize vals with flat and idx_out with original indices
        vals[:N] = flat
        idx_out[:N] = torch.arange(N, device=device)
        # Launch stable bitonic argsort
        stable_bitonic_argsort[(1,)](vals, idx_out, N, BLOCK_SORT, LOG_SORT)

        # Extract sorted_token_indices (first N)
        sorted_token_indices = idx_out[:N].to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
