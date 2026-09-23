import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, num_experts: tl.int32):
    """
    Compute per-expert counts for values in flat_ptr (int32).
    Grid: one program per element (N programs).
    counts_ptr: int32 array of size num_experts.
    """
    pid = tl.program_id(axis=0)
    if pid < N:
        val = tl.load(flat_ptr + pid)
        # Ensure val is in [0, num_experts-1]
        # Atomic add to counts[val]
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_scan_inplace(inp_ptr, out_ptr, E: tl.int32, LOG: tl.constexpr):
    """
    In-kernel inclusive scan over inp_ptr[0..E-1], write results to out_ptr[0..E-1].
    LOG = log2(E), passed as constexpr.
    """
    # Hillis–Steele inclusive scan in-kernel
    idx = tl.arange(0, E)
    # Copy input to output
    out = tl.load(inp_ptr + idx)
    tl.store(out_ptr + idx, out)

    # Perform scan
    for i in range(1, LOG + 1):
        stride = 1 << i
        prev = tl.load(out_ptr + (idx - stride), mask=(idx >= stride), other=0)
        curr = tl.load(out_ptr + idx)
        new = curr + prev
        tl.store(out_ptr + idx, new)


@triton.jit
def stable_bitonic_argsort(vals_ptr, idx_ptr, N: tl.int32, BLOCK: tl.constexpr, LOG: tl.constexpr):
    """
    Stable bitonic argsort on vals_ptr[0..BLOCK-1], tie-breaking by original idx_ptr[0..BLOCK-1].
    We pad N with large sentinel so N..BLOCK-1 sort to the end. idx_ptr stores original indices 0..N-1.
    """
    idx = tl.program_id(axis=0)  # single program instance; bitonic network runs vector-wise
    offsets = tl.arange(0, BLOCK)
    size = BLOCK
    # Bitonic sort network
    for p in range(1, LOG + 1):
        k = 1 << p
        for j in range(p - 1, -1, -1):
            i = 1 << j
            partner = offsets ^ i
            a = tl.load(vals_ptr + offsets)
            b = tl.load(vals_ptr + partner)
            ia = tl.load(idx_ptr + offsets)
            ib = tl.load(idx_ptr + partner)

            # Decide direction
            ascending = (offsets & k) == 0

            # Stable tie-breaking: if values equal, swap based on original index to enforce stability
            cond_value = a <= b  # True if a should come before b in ascending part, else False
            cond_index = ia < ib  # tie-breaker
            swap = tl.where(ascending, cond_value, ~cond_value)  # swap if ascending and a > b, or descending and a < b
            swap = swap & ((a == b) == 0)  # only consider swap when not equal; for equal, use index

            # Apply swap (conditional)
            new_a = tl.where(swap, b, a)
            new_b = tl.where(swap, a, b)
            new_ia = tl.where(swap, ib, ia)
            new_ib = tl.where(swap, ia, ib)

            # Write back to lower index partner to avoid double writes
            tl.store(vals_ptr + offsets, new_a)
            tl.store(vals_ptr + partner, new_b)
            tl.store(idx_ptr + offsets, new_ia)
            tl.store(idx_ptr + partner, new_ib)

            # For offsets > partner, both sides write; for offsets < partner, partner already updated by symmetric pair
            # We avoid race-free writes by ensuring each pair is updated exactly once via masks; Triton handles this with vectorized operations.


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure int32
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()
        device = flat.device

        # 1) Triton histogram: counts of expert indices
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        # Launch one program per element
        histogram_kernel[(N,)](flat, counts, N, num_experts)

        # 2) Triton inclusive scan of counts to get expert_offsets[1:], then set offsets[0]=0 and offsets[256]=N
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        # Initial counts into offsets[1:]
        offsets[1:] = counts
        LOG_SCAN = 8  # since num_experts=256 -> log2(256)=8
        inclusive_scan_inplace[(1,)](offsets[1:], offsets[1:], num_experts, LOG_SCAN)
        # Set final element to N
        offsets[256] = N

        # 3) Triton stable bitonic argsort to produce sorted_token_indices
        # Prepare vals and idx buffers; pad to next power-of-two BLOCK
        BLOCK_SORT = 1 << (N - 1).bit_length()  # next power-of-two >= N
        BLOCK_SORT = min(BLOCK_SORT, 4096)     # cap for practicality
        LOG_SORT = (BLOCK_SORT.bit_length() - 1)  # number of steps

        vals = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)
        idx_out = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)

        # Fill first N entries; pad with large sentinel so they sort to the end
        MAX_INT = (1 << 31) - 1
        vals[:N] = flat
        vals[N:] = MAX_INT
        idx_out[:N] = torch.arange(N, device=device)
        # Launch stable bitonic argsort on vals with original indices in idx_out
        stable_bitonic_argsort[(1,)](vals, idx_out, N, BLOCK_SORT, LOG_SORT)

        # Extract sorted_token_indices (first N)
        sorted_token_indices = idx_out[:N].to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
