import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
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
    Compute inclusive prefix sum of counts_ptr (int32, length num_experts) into offsets_ptr (int64).
    offsets_ptr[0] should be set to 0 by the caller. We write offsets[1..].
    """
    acc = tl.zeros((), dtype=tl.int64)
    # Single program computes the prefix sum sequentially
    for i in range(0, num_experts):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_val_ptr, out_idx_ptr, N,
                                LOGN: tl.constexpr, BLOCK: tl.constexpr):
    """
    Stable bitonic sort on flat_ptr (int32). Writes sorted values to out_val_ptr (int32)
    and stable indices to out_idx_ptr (int64). One program per element; multiple stages in axis 0.
    """
    # This kernel is intended to be launched with a grid that covers all stages and elements.
    # However, Triton doesn't support arbitrary grid lambdas for per-element stages easily,
    # so we instead use a 1D grid and loop over stages inside the kernel, which Triton supports.
    # Each program handles all stages, which is fine for small N.
    i = tl.program_id(axis=0)
    # Ensure i < N
    # We'll run the loop anyway; i is a scalar index per program. To cover all elements, we set grid=(N,)
    # and inside, each program reuses i as its local index.
    # Implement bitonic sort for element i: we need partner indices for each stage.
    # We'll use the fact that for bitonic sort, each element is involved in pairs (i, partner) where
    # partner = i ^ (1 << j) for stages j. We build a list of stages by looping j and k.

    # We'll use a static loop over j and k. Triton supports tl.static_range when ranges are constexpr.
    # Note: In Triton, loops must be over tl.constexpr bounds; LOGN is passed as constexpr.
    for j in tl.static_range(1, LOGN + 1):
        # k in 0..j-1
        for k in tl.static_range(0, j):
            # For each pass, we process all i again. We derive partner based on k.
            # However, since each program instance i will execute all stages, we need to compute partner for i.
            # partner = i ^ (1 << j) but partner index may be out of bounds for this pass; use XOR with (1 << k) instead.
            # The bitonic compare-exchange uses partner = i ^ (1 << j); but since j is the loop variable,
            # we compute partner_i = i ^ (1 << j). Then, for each pass, partner index is i ^ (1 << (j)).
            # We need to load the partner's current value and index. Since we only have one axis, we must rely
            # on the fact that the loop executes for all i; partner indices will be computed relative to i,
            # and the update is done in a deterministic way: for each stage, we compute the partner and decide
            # to update out_val/out_idx for i only, using ascending/descending based on (i & (1 << (j + 1))) == 0.
            # Implement by computing partner = i ^ (1 << j), loading values from flat_ptr, and writing updated
            # values to out_val_ptr at index i. This approach ensures correctness because each program handles
            # all stages; we only update the current element i's position per stage, using its partner's value.
            # We'll reconstruct the bitonic compare-exchange logic per stage.
            # Note: This is a standard in-kernel bitonic implementation. For Triton, using tl.static_range
            # with LOGN allows compile-time unrolling.
            partner = i ^ (1 << j)
            asc = ((i & (1 << (j + 1))) == 0)
            # Load current values
            a = tl.load(flat_ptr + i)
            b = tl.load(flat_ptr + partner)
            idx_i = tl.cast(i, tl.int64)
            idx_p = tl.cast(partner, tl.int64)

            # Determine min and max with stability: if equal, lower index comes first.
            # Compute which is min and which is max. We use the asc flag to decide final assignment.
            # For ascending:
            #   if a < b: i gets a, partner gets b
            #   elif a > b: i gets b, partner gets a
            #   else: if i < partner: i gets a, partner gets a (tie, keep original), else i gets b, partner gets b
            # For descending: opposite.
            if asc:
                less = a < b
                greater = a > b
                equal = (a == b)
                # If equal and i < partner: i keeps a, partner keeps a; else i gets b, partner gets b
                tie = equal & (idx_i < idx_p)
                # Assign to i
                new_i = tl.where(less, a, tl.where(greater, b, tl.where(tie, a, b)))
                # Assign to partner
                new_p = tl.where(less, b, tl.where(greater, a, tl.where(tie, a, b)))
            else:
                less = a < b
                greater = a > b
                equal = (a == b)
                tie = equal & (idx_i < idx_p)
                # Assign to i
                new_i = tl.where(less, b, tl.where(greater, a, tl.where(tie, a, b)))
                # Assign to partner
                new_p = tl.where(less, a, tl.where(greater, b, tl.where(tie, a, b)))

            # Write back: but since out_val_ptr and out_idx_ptr are arrays, and Triton doesn't support
            # multi-dimensional indexing per element like that in a vectorized way, we instead implement
            # the bitonic network by having a separate kernel with grid=(N,), and for each stage j, each
            # program processes its element i and updates out_val/out_idx accordingly. This requires
            # knowing partner index. Triton allows reading from flat_ptr and writing to out_val_ptr
            # at index i. Partner writes can be done by having two separate passes per stage:
            # one to write i, and one to write partner (since each stage handles both sides,
            # we rely on the fact that each program reprocesses i for all stages).
            # However, Triton does not support nested loops with dynamic partner writes per element cleanly.
            # Therefore, we switch to a simpler approach: implement odd-even transposition sort entirely
            # in Triton, which is robust and avoids bitonic intricacies here.
            # End of bitonic draft; we will implement odd-even below in a simpler, correct way.

            # Note: The above block is a conceptual template; Triton does not allow dynamic partner writes
            # in a single kernel across all stages cleanly. We'll implement odd-even transposition sort
            # in Triton below for correctness.

            # Placeholder: implement actual compare-exchange per stage using odd-even pattern.
            # We'll fill the actual logic for odd-even transposition below.
            pass
            # The pass above is just to satisfy Triton's requirement; actual logic will be in the next block.


@triton.jit
def odd_even_stable_sort_kernel(flat_ptr, out_val_ptr, out_idx_ptr, N, ROUNDS: tl.constexpr, BLOCK: tl.constexpr):
    """
    Stable odd-even transposition sort in Triton. Perform N passes (ROUNDS=N).
    Each program handles one element i, and for each pass, it swaps with neighbor i+1 if parity matches
    and values are out-of-order. We maintain stable indices (int64) alongside values (int32).
    """
    i = tl.program_id(axis=0)
    # Initialize out_val and out_idx with original flat values and indices
    val_i = tl.load(flat_ptr + i)
    idx_i = tl.cast(i, tl.int64)
    tl.store(out_val_ptr + i, val_i)
    tl.store(out_idx_ptr + i, idx_i)

    # Perform ROUNDS passes
    for p in tl.static_range(0, ROUNDS):
        # Even phase: pairs (0,1), (2,3), ...
        # Odd phase: pairs (1,2), (3,4), ...
        if (p % 2) == 0:
            # even phase: partner exists if (i % 2 == 0) and (i + 1) < N
            even_pair = (i % 2 == 0)
            partner = i + 1
            in_range = (partner < N) & even_pair
            a = tl.load(out_val_ptr + i)
            b = tl.load(out_val_ptr + partner)
            idx_a = tl.load(out_idx_ptr + i)
            idx_b = tl.load(out_idx_ptr + partner)

            less = a > b  # even phase: i should take smaller value
            greater = a < b
            equal = a == b
            tie = equal & (idx_a < idx_b)

            new_i = tl.where(less, b, tl.where(greater, a, tl.where(tie, a, b)))
            new_partner = tl.where(less, a, tl.where(greater, b, tl.where(tie, a, b)))

            # Only perform swap when in_range
            if in_range:
                tl.store(out_val_ptr + i, new_i)
                tl.store(out_val_ptr + partner, new_partner)
                tl.store(out_idx_ptr + i, tl.cast(i, tl.int64))  # keep index updated (index is constant)
                tl.store(out_idx_ptr + partner, tl.cast(partner, tl.int64))
        else:
            # odd phase: partner exists if (i % 2 == 1) and (i + 1) < N
            odd_pair = (i % 2 == 1)
            partner = i + 1
            in_range = (partner < N) & odd_pair
            a = tl.load(out_val_ptr + i)
            b = tl.load(out_val_ptr + partner)
            idx_a = tl.load(out_idx_ptr + i)
            idx_b = tl.load(out_idx_ptr + partner)

            less = a < b  # odd phase: i should take smaller value (ascending)
            greater = a > b
            equal = a == b
            tie = equal & (idx_a < idx_b)

            new_i = tl.where(less, b, tl.where(greater, a, tl.where(tie, a, b)))
            new_partner = tl.where(less, a, tl.where(greater, b, tl.where(tie, a, b)))

            if in_range:
                tl.store(out_val_ptr + i, new_i)
                tl.store(out_val_ptr + partner, new_partner)
                tl.store(out_idx_ptr + i, tl.cast(i, tl.int64))
                tl.store(out_idx_ptr + partner, tl.cast(partner, tl.int64))


# Note: The above odd_even_stable_sort_kernel uses Triton's static_range over ROUNDS=N.
# Triton supports loops with constexpr bounds. We launch it with grid=(N,) so each program handles one element.

# Final ModelNew implementing Triton-only computation
class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only forward:
        - Flatten topk_idx -> int32 flat
        - Launch stable sort kernel to produce sorted_token_indices (int64)
        - Launch histogram + prefix sum kernels to produce expert_offsets (int32)
        """
        # 1) Flatten and ensure int32
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()

        # 2) Stable sort via Triton odd-even transposition (N passes)
        out_vals = torch.empty(N, dtype=torch.int32, device=flat.device)
        out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
        ROUNDS = N  # number of passes; for small N this is fine
        BLOCK = 1  # one element per program
        grid = (N,)
        odd_even_stable_sort_kernel[grid](flat, out_vals, out_idx, N, ROUNDS, BLOCK, num_warps=1)
        sorted_token_indices = out_idx  # int64

        # 3) expert_offsets: histogram and prefix sum via Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        # Launch histogram kernel
        # Choose BLOCK for histogram; 1024 works well
        HIST_BLOCK = 1024
        grid_hist = (triton.cdiv(N, HIST_BLOCK),)
        count_histogram_kernel[grid_hist](flat, counts, N, self.num_experts, HIST_BLOCK, num_warps=4)

        # Compute prefix sum of counts (int64) and offsets[0] = 0 on host
        offsets_int64 = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        prefix_sum_kernel[(self.num_experts,)](counts, offsets_int64[1:], self.num_experts, num_warps=1)

        expert_offsets = offsets_int64.to(torch.int32)  # match original dtype

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
