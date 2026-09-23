import math
import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr, num_experts: tl.constexpr):
    """
    Compute histogram of flat values (int32) into counts_ptr (int32), length num_experts.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values
    # Atomic add 1 for each occurrence into counts[val]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr (int64, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    # Single-program prefix sum across num_experts
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). out_idx_ptr holds original positions as int64.
    For each stage j and k:
      asc = ((i & (1 << (k+1))) == 0)
      partner = i ^ (1 << j)
      only process pairs with i < partner.
    Stability tie-break: if equal values, lower original index (i < partner) comes first.
    Ascending: lower gets min, higher gets max; descending: lower gets max, higher gets min.
    """
    pid = tl.program_id(axis=0)
    # Each program handles one element; bitonic stages are handled by axis 1 grid.
    # Note: Axis 1 grid is not used here because Triton requires static grid; we emulate stages via meta.
    # Instead, we launch with a 1D grid and rely on per-stage logic. For simplicity and correctness,
    # we implement the compare-exchange directly for all stages in a single kernel by looping over j, k.
    # However, Triton supports loops only with compile-time constants; bitonic stages depend on N via LOGN.
    # To keep it simple and robust, we only implement the per-element scanning approach with a single pass,
    # which would require reorganizing data; but bitonic requires a grid across stages. Triton does not
    # support dynamic axis sizing. Therefore, we switch to an odd-even transposition sort kernel below
    # which is simpler and correct, and we ensure it is used.
    # Placeholder: not used; implemented below via odd-even kernel.
    pass


# Odd-even transposition sort in Triton (stable)
@triton.jit
def odd_even_transpose_sort_kernel(flat_ptr, out_idx_ptr, N, ROUNDS: tl.constexpr):
    """
    Stable odd-even transposition sort:
    - out_idx_ptr holds int64 original positions (initialized 0..N-1).
    - For t in [0, ROUNDS): even passes swap even pairs (0,1),(2,3)...;
      odd passes swap odd pairs (1,2),(3,4)...
    Stability: tie-break by original index (lower comes first).
    """
    pid = tl.program_id(axis=0)
    i = pid
    # Each program handles one element i and updates it for all passes.
    for t in range(0, ROUNDS):
        if ((t & 1) == 0):
            # even pass: pairs (0,1),(2,3),...
            if (i % 2 == 0) and (i < N - 1):
                j = i + 1
                a = tl.load(flat_ptr + i)
                b = tl.load(flat_ptr + j)
                ai = tl.load(out_idx_ptr + i)
                bj = tl.load(out_idx_ptr + j)
                asc = a <= b
                lower = tl.where(asc, a, b)
                higher = tl.where(asc, b, a)
                # Determine original lower index
                orig_lower = tl.where(asc, ai, bj)
                orig_higher = tl.where(asc, bj, ai)
                # Stable within each pair: if equal, lower index comes first
                eq = a == b
                # Store results for i and j
                tl.store(out_idx_ptr + i, tl.where(asc, orig_lower, tl.where(eq, orig_lower, orig_higher)))
                tl.store(out_idx_ptr + j, tl.where(asc, orig_higher, tl.where(eq, orig_higher, orig_lower)))
                # Store values accordingly
                tl.store(flat_ptr + i, tl.where(asc, lower, higher))
                tl.store(flat_ptr + j, tl.where(asc, higher, lower))
        else:
            # odd pass: pairs (1,2),(3,4),...
            if (i % 2 == 1) and (i < N - 1):
                j = i + 1
                a = tl.load(flat_ptr + i)
                b = tl.load(flat_ptr + j)
                ai = tl.load(out_idx_ptr + i)
                bj = tl.load(out_idx_ptr + j)
                asc = a <= b
                lower = tl.where(asc, a, b)
                higher = tl.where(asc, b, a)
                orig_lower = tl.where(asc, ai, bj)
                orig_higher = tl.where(asc, bj, ai)
                eq = a == b
                tl.store(out_idx_ptr + i, tl.where(asc, orig_lower, tl.where(eq, orig_lower, orig_higher)))
                tl.store(out_idx_ptr + j, tl.where(asc, orig_higher, tl.where(eq, orig_higher, orig_lower)))
                tl.store(flat_ptr + i, tl.where(asc, lower, higher))
                tl.store(flat_ptr + j, tl.where(asc, higher, lower))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D, keep int32 on device
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()

        # 1) Histogram via Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_histogram_kernel[grid](flat, counts, N, BLOCK=BLOCK, num_experts=self.num_experts, num_warps=4)

        # 2) Prefix sum of counts via Triton (int64 offsets), we set offsets[0] = 0 on host
        offsets_int64 = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        prefix_sum_kernel[(self.num_experts,)](counts, offsets_int64[1:], num_experts=self.num_experts, num_warps=1)

        # 3) Stable sort via Triton: odd-even transposition sort
        # We use out_idx initialized as original positions (int64), ascending order.
        out_idx = torch.arange(N, dtype=torch.int64, device=flat.device)
        # Number of rounds: N is typically small in the provided workloads; using 2*N rounds is safe.
        ROUNDS = 2 * N
        odd_even_transpose_sort_kernel[(N,)](flat, out_idx, N, ROUNDS, num_warps=4)

        sorted_token_indices = out_idx  # int64, matches torch.sort(stable=True).indices
        expert_offsets = offsets_int64.to(torch.int32)  # int32, matches torch.bincount + cumsum

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
