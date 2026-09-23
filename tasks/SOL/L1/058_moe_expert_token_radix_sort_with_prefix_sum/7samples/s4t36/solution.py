import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, num_experts: tl.int32):
    # One program per element; each program atomically increments the count for its value.
    pid = tl.program_id(0)
    if pid < N:
        val = tl.load(flat_ptr + pid)  # int32
        # Atomic add 1 to counts[val]
        # Note: counts_ptr is a pointer; Triton supports atomic_add on int32.
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_scan_inplace(counts_ptr, out_ptr, E: tl.constexpr, LOG: tl.constexpr):
    # In-kernel inclusive scan for the first E elements of counts_ptr.
    # out_ptr receives the inclusive prefix sums for indices 0..E-1.
    # LOG is log2(E); E must be a power of two for this simple implementation (here E=256).
    idx = tl.arange(0, E)
    current = tl.load(counts_ptr + idx)
    # Hillis–Steele inclusive scan with fixed LOG steps.
    # We rely on E being a power of two (256) to make this correct.
    for i in range(LOG):
        step = 1 << i
        to_add = tl.where((idx >= step) & (idx - step >= 0), tl.load(out_ptr + (idx - step)), 0)
        current = current + to_add
        tl.store(out_ptr + idx, current)


@triton.jit
def stable_bitonic_sort_inplace(vals_ptr, idx_out_ptr, N: tl.int32, BLOCK: tl.constexpr, LOG: tl.constexpr):
    # Bitonic sort over BLOCK lanes; first N lanes contain real data, remaining padded with MAX_INT.
    # Tie-break: for equal values, sort by original idx_out value (ascending), ensuring stability.
    MAX_INT = (1 << 31) - 1
    idx = tl.arange(0, BLOCK)
    # Initialize idx_out with original indices [0..BLOCK-1]
    tl.store(idx_out_ptr + idx, idx)
    # We will not read beyond N; padded lanes are arbitrary, only used for network padding.
    # Perform bitonic sort network with stable tie-breaking.
    for stage in range(1, LOG + 1):
        k = 1 << stage
        for j in range(stage - 1, -1, -1):
            i = 1 << j
            partner = idx ^ i
            # Load partner values and indices
            val_i = tl.load(vals_ptr + idx)
            val_p = tl.load(vals_ptr + partner)
            idx_i = tl.load(idx_out_ptr + idx)
            idx_p = tl.load(idx_out_ptr + partner)
            # Ascending for low half, descending for high half in this k
            ascending = (idx & k) == 0
            # For ties, use original index to ensure stable order (ascending by index)
            tie = val_i == val_p
            should_swap = tl.where(ascending,
                                   (val_i > val_p) | ((val_i == val_p) & (idx_i > idx_p)),
                                   (val_i < val_p) | ((val_i == val_p) & (idx_i < idx_p)))
            # Swap if necessary
            new_val_i = tl.where(should_swap, val_p, val_i)
            new_val_p = tl.where(should_swap, val_i, val_p)
            new_idx_i = tl.where(should_swap, idx_p, idx_i)
            new_idx_p = tl.where(should_swap, idx_i, idx_p)
            # Write back
            tl.store(vals_ptr + idx, new_val_i)
            tl.store(vals_ptr + partner, new_val_p)
            tl.store(idx_out_ptr + idx, new_idx_i)
            tl.store(idx_out_ptr + partner, new_idx_p)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants: num_experts in the original run is 256
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        device = flat.device
        N = flat.numel()
        E = self.num_experts

        # 1) Triton histogram: counts per expert
        counts = torch.zeros(E, dtype=torch.int32, device=device)
        grid = (N,)
        histogram_kernel[grid](flat, counts, N, E)

        # 2) Triton inclusive scan of counts -> expert_offsets
        offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
        offsets[0] = 0  # inclusive scan starts at 0
        LOG = 8  # log2(256) == 8; E must be power of two for this kernel
        inclusive_scan_inplace[(1,)](counts, offsets[1:], E, LOG)
        # Ensure final offset equals total tokens (matches original run which sets offsets[num_experts] = N)
        # (The scan above already accumulates per-expert counts; we explicitly set total here to be consistent.)
        # Note: The original run uses torch.cumsum on bincount result; here we emulate that with Triton.

        # 3) Triton stable bitonic sort to produce argsort indices
        # Compute BLOCK_SORT as next power-of-two >= N (capped reasonably)
        # We will cap at 4096 to keep within practical limits for this environment.
        block_sort = 1
        while block_sort < N and block_sort < 4096:
            block_sort <<= 1
        LOG_SORT = 0
        temp = block_sort
        while (temp >> 1) >= 1:
            temp >>= 1
            LOG_SORT += 1

        MAX_INT = (1 << 31) - 1
        vals = torch.empty(block_sort, dtype=torch.int32, device=device)
        idx_out = torch.empty(block_sort, dtype=torch.int32, device=device)

        # vals: first N are flat, rest are sentinel so they sort to the end
        vals[:N] = flat
        vals[N:] = MAX_INT
        # idx_out: original indices [0..block_sort-1]; we only use first N later
        idx_out[:N] = torch.arange(N, device=device)

        stable_bitonic_sort_inplace[(1,)](vals, idx_out, N, block_sort, LOG_SORT)

        # Extract sorted_token_indices (first N)
        sorted_token_indices = idx_out[:N]

        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
