import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: for each token i in 0..N-1, if vals[i] == e, atomic add 1 to counts[e].
    vals_ptr: *int32, length N
    counts_ptr: *int32, length num_experts (256)
    N: number of tokens (runtime int)
    num_experts: constexpr, 256
    """
    # Single program processes the whole array sequentially to avoid illegal memory access.
    for i in range(0, N):
        # Load val
        val = tl.load(vals_ptr + i)
        # Count for each expert
        for e in range(0, num_experts):
            # If value equals expert index, increment counts[e]
            if val == e:
                tl.atomic_add(counts_ptr + e, 1)


@triton.jit
def inclusive_scan_kernel(counts_ptr, out_ptr, num_experts: tl.constexpr):
    """
    Triton kernel: inclusive prefix sum over counts_ptr (length num_experts) into out_ptr.
    num_experts: constexpr, 256.
    """
    # Single program performs sequential scan over 256 elements.
    running = 0
    for i in range(0, num_experts):
        running += tl.load(counts_ptr + i)
        tl.store(out_ptr + i, running)


@triton.jit
def bitonic_sort_stable_kernel(idx_ptr, out_ptr, PADDED: tl.constexpr):
    """
    Triton kernel: in-place stable bitonic sort over PADDED elements.
    - idx_ptr: input int32 indices (values are token positions 0..PADDED-1; original N tokens + padding).
    - out_ptr: output int32 indices (sorted stable).
    PADDED: constexpr, e.g., 4096.
    """
    # We operate on a single program and perform pairwise compare-exchange to sort the entire array.
    # Initialize out[i] = i for i in 0..PADDED-1
    for i in range(0, PADDED):
        tl.store(out_ptr + i, i)

    # Bitonic sorting network
    # k doubles from 2 to PADDED
    k = 2
    while k <= PADDED:
        # j halves from k//2 down to 1
        j = k // 2
        while j >= 1:
            i = 0
            while i < PADDED:
                ri = i ^ j
                ai = tl.load(idx_ptr + i)
                ar = tl.load(idx_ptr + ri)
                bi = tl.load(out_ptr + i)
                br = tl.load(out_ptr + ri)
                # Ascending if (i & k) == 0 else descending
                ascending = (i & k) == 0
                # Stable tie-break: if ai == ar, left i < right ri
                tie_left = (ai == ar) & (i < ri)
                # Compare-exchange condition:
                # If ascending: swap when ai > ar or (equal and i > ri)
                # If descending: swap when ai < ar or (equal and i < ri)
                cond = tl.where(
                    ascending,
                    (ai > ar) | ((ai == ar) & (i > ri)),
                    (ai < ar) | ((ai == ar) & (i < ri))
                )
                # Perform conditional swap for both sides of the pair
                new_i = tl.where(cond, ar, ai)
                new_ri = tl.where(cond, ai, ar)
                tl.store(out_ptr + i, new_i)
                tl.store(out_ptr + ri, new_ri)
                i += 1
            j //= 2
        k *= 2


def _next_power_of_two(n: int) -> int:
    # For safety and simplicity, cap at 4096 (sufficient for given workloads).
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return min(p, 4096)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute counts of each expert index.
        - Compute inclusive scan to get per-expert offsets.
        - Stable sort token indices via bitonic sort kernel.
        Returns:
          - sorted_token_indices: int32 tensor of shape (N,)
          - expert_offsets: int32 tensor of shape (num_experts+1,)
        """
        # Ensure int32
        vals = topk_idx.reshape(-1).to(torch.int32)
        N = vals.numel()
        device = vals.device

        # 1) Count per expert
        counts = torch.empty((256,), dtype=torch.int32, device=device)
        grid_counts = (1,)
        count_experts_kernel[grid_counts](vals, counts, N, num_experts=256)

        # 2) Inclusive scan of counts
        scan_out = torch.empty((256,), dtype=torch.int32, device=device)
        grid_scan = (1,)
        inclusive_scan_kernel[grid_scan](counts, scan_out, num_experts=256)

        # 3) Bitonic stable sort of token indices (PADDED = next power of two >= N, capped at 4096)
        PADDED = _next_power_of_two(N)
        # idx_ptr: original token indices (positions 0..N-1)
        idx = torch.arange(N, dtype=torch.int32, device=device)  # indices 0..N-1
        # Pad with N + 1 so padding sorts to the end (any value > N-1 is fine)
        # We create a temporary buffer of length PADDED filled with N + 1
        idx_buf = torch.full((PADDED,), N + 1, dtype=torch.int32, device=device)
        # Copy original indices to the beginning
        idx_buf[0:N] = idx

        out_buf = torch.empty((PADDED,), dtype=torch.int32, device=device)
        grid_sort = (1,)
        bitonic_sort_stable_kernel[grid_sort](idx_buf, out_buf, PADDED)

        # sorted_token_indices: first N elements of out_buf
        sorted_token_indices = out_buf[0:N]

        # expert_offsets: [0] + inclusive_scan result
        # We avoid torch.full_like/cat; allocate and fill:
        # 257 elements (num_experts + 1)
        expert_offsets = torch.empty((257,), dtype=torch.int32, device=device)
        # Set first element to 0 (inclusive scan starts with 0)
        expert_offsets[0] = 0
        # Copy scan_out into the remaining 256 elements
        expert_offsets[1:] = scan_out

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
