import torch
import triton
import triton.language as tl


# Kernel: For each expert id e in [0..NUM_EXPERTS), count occurrences in flat.
@triton.jit
def counts_kernel(flat_ptr, counts_ptr, M, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    # Initialize count
    count = tl.zeros((), dtype=tl.int32)
    # Iterate over flat in chunks of BLOCK
    for j in range(0, M, BLOCK):
        offs = j + tl.arange(0, BLOCK)
        mask = offs < M
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        eq = (vals == e) & mask
        count += tl.sum(eq.to(tl.int32), axis=0)
    # Store count for this expert
    tl.store(counts_ptr + e, count)


# Kernel: Inclusive prefix sum over counts[0..NUM_EXPERTS-1] -> offsets_incl[0..NUM_EXPERTS-1]
@triton.jit
def scan_inclusive_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    running = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        cnt = tl.load(counts_ptr + i)
        running += cnt
        tl.store(offsets_ptr + i, running)


# Kernel: Finalize expert_offsets: write offsets_incl[:NUM_EXPERTS] and set last = total_count + 1.
@triton.jit
def finalize_offsets_kernel(offsets_incl_ptr, total_count_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    # Copy inclusive prefix sums to offsets[:NUM_EXPERTS]
    for e in range(0, NUM_EXPERTS):
        tl.store(offsets_ptr + e, tl.load(offsets_incl_ptr + e))
    # Set last element = total_count + 1
    total = tl.load(total_count_ptr)
    tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


# Kernel: Compute stable permutation indices. Writes sorted_token_indices (int64).
@triton.jit
def stable_permutation_kernel(flat_ptr, offsets_ptr, sorted_idx_ptr, M, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    # We implement stable sort by assigning base offsets to each key and writing to out at those positions.
    # This kernel assigns each original index j its sorted position.
    # Use a per-key loop and then per-j chunk loop to ensure deterministic behavior.
    for e in range(0, NUM_EXPERTS):
        base = tl.load(offsets_ptr + (e - 1))  # base_excl for key e; if e==0, base=0
        # Process all j in chunks; write to sorted_idx[j] = base
        for j in range(0, M, BLOCK):
            offs = j + tl.arange(0, BLOCK)
            mask = offs < M
            vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 keys
            is_e = (vals == e) & mask
            # Where is_e is True, write base (as int64) to sorted_idx at position offs
            base64 = base.to(tl.int64)
            # Build int64 zeros for non-matching positions
            zero64 = tl.zeros([BLOCK], dtype=tl.int64)
            out = tl.where(is_e, base64, zero64)
            tl.store(sorted_idx_ptr + offs, out, mask=mask)


# Helper Triton kernel to compute total_count = sum(counts) and store to total_count_ptr.
@triton.jit
def reduce_sum_counts_kernel(counts_ptr, total_count_ptr, NUM_EXPERTS: tl.constexpr):
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        total += tl.load(counts_ptr + i)
    tl.store(total_count_ptr, total)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D, ensure int32 for Triton loads
        flat = topk_idx.reshape(-1)
        M = flat.numel()
        # Ensure flat is on same device and int32
        if not flat.is_contiguous():
            flat = flat.contiguous()
        flat_i32 = flat.to(torch.int32)

        device = flat_i32.device
        NUM_EXPERTS = self.num_experts

        # 1) Launch counts_kernel: counts[e] = number of occurrences of e in flat
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        # Launch 1 program per expert
        counts_kernel[(NUM_EXPERTS,)](flat_i32, counts, M, NUM_EXPERTS, BLOCK=1024, num_warps=4)

        # 2) Launch scan_inclusive_kernel: offsets_incl[e] = inclusive prefix sum
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        scan_inclusive_kernel[(1,)](counts, offsets_incl, NUM_EXPERTS, num_warps=1)

        # 3) Compute total_count using Triton reduction (avoid torch.sum)
        total_count_ptr = torch.empty(1, dtype=torch.int32, device=device)
        reduce_sum_counts_kernel[(1,)](counts, total_count_ptr, NUM_EXPERTS, num_warps=1)

        # 4) Launch finalize_offsets_kernel to produce expert_offsets of length NUM_EXPERTS + 1
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        finalize_offsets_kernel[(1,)](offsets_incl, total_count_ptr, expert_offsets, NUM_EXPERTS, num_warps=1)

        # 5) Launch stable_permutation_kernel to compute sorted_token_indices (int64)
        sorted_idx = torch.empty(M, dtype=torch.int64, device=device)
        stable_permutation_kernel[(1,)](flat_i32, offsets_incl, sorted_idx, M, NUM_EXPERTS, BLOCK=1024, num_warps=4)

        return sorted_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
