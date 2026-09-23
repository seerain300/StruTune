import torch
import triton
import triton.language as tl


# Triton kernel: stable argsort of flat into out (permutation indices).
# For each position j:
#   key_j = flat[j]
#   base_excl_prev = base_excl[key_j - 1] if key_j > 0 else 0
#   tie_j = count of earlier elements t < j with same key and flat[t] < flat[j] (stable tie-break)
#   rank_j = base_excl_prev + tie_j
#   out[j] = rank_j
# Then update per-key base_excl and tie_rank:
#   for e in [0..255]: if e < key_j: base_excl[e] += 1; if e == key_j: tie_rank[e] += 1
@triton.jit
def stable_argsort_kernel(
    flat_ptr,            # *const int32, length M
    out_ptr,             # *int32, length M (output permutation indices)
    base_excl_ptr,       # *int32, length 256 (exclusive prefix per key)
    tie_rank_ptr,        # *int32, length 256 (stable tie counter per key)
    M,                   # int32, total number of elements
    NUM_EXPERTS: tl.constexpr,  # compile-time constant 256
    BLOCK_SIZE: tl.constexpr     # e.g., 1024
):
    j = tl.program_id(axis=0)
    mask_j = j < M

    # Load current value at position j
    val_j = tl.load(flat_ptr + j, mask=mask_j, other=0)  # int32

    # Compute base_excl for key_j-1 (primary ordering)
    base_prev = tl.zeros((), dtype=tl.int32)
    if val_j > 0:
        base_prev = tl.load(base_excl_ptr + (val_j - 1))

    # Compute tie_j: number of elements before j with same key and smaller value
    # Loop over t from 0..j-1 in chunks and accumulate tie_j.
    tie_j = tl.zeros((), dtype=tl.int32)
    # Simple loop over t in chunks; j is scalar here, so we iterate t from 0..j-1 in chunks
    t = 0
    while t < j:
        offs = t + tl.arange(0, BLOCK_SIZE)
        mask_t = offs < j
        vals_t = tl.load(flat_ptr + offs, mask=mask_t, other=0)
        # Count elements equal to key_j and less than val_j
        is_equal = vals_t == val_j
        is_less = vals_t < val_j
        tie_j += tl.sum(is_equal & mask_t & is_less, axis=0)
        t += BLOCK_SIZE

    rank_j = base_prev + tie_j
    tl.store(out_ptr + j, rank_j, mask=mask_j)

    # Update base_excl and tie_rank for key=val_j
    # For e < val_j: base_excl[e] += 1
    # For e == val_j: tie_rank[e] += 1
    for e in range(NUM_EXPERTS):
        base_e = tl.load(base_excl_ptr + e)
        tie_e = tl.load(tie_rank_ptr + e)
        if e < val_j:
            new_base = base_e + 1
        else:
            new_base = base_e
        new_tie = tie_e
        if e == val_j:
            new_tie = tie_e + 1
        tl.store(base_excl_ptr + e, new_base)
        tl.store(tie_rank_ptr + e, new_tie)


# Triton kernel: compute per-key counts via atomic reduction.
# For each element j, if j < M, increment counts[flat[j]] and total_count.
@triton.jit
def _count_per_key_atomic_kernel(
    flat_ptr,            # *const int32, length M
    counts_ptr,          # *int32, length 256
    total_ptr,           # *int32, scalar
    M,                   # int32
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Iterate over elements in chunks; each program processes a chunk and atomically adds into counts.
    # Launch with a single program for simplicity since M is a single scalar grid; we can loop inside.
    # But Triton expects grid size; we launch with grid=(1,) and iterate over M inside.
    start = 0
    while start < M:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < M
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)
        # Accumulate counts per key in this chunk
        for jj in range(BLOCK_SIZE):
            valid = start + jj < M
            val_elem = vals[jj] if valid else 0
            if valid:
                tl.atomic_add(counts_ptr + val_elem, 1)
        start += BLOCK_SIZE


# Triton kernel: inclusive prefix sums of counts (NUM_EXPERTS=256, small).
@triton.jit
def _inclusive_scan_counts_kernel(
    counts_ptr,          # *int32, length 256
    incl_ptr,            # *int32, length 256
    NUM_EXPERTS: tl.constexpr
):
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(NUM_EXPERTS):
        acc += tl.load(counts_ptr + i)
        tl.store(incl_ptr + i, acc)


def _run_triton_sorted_indices(flat: torch.Tensor) -> torch.Tensor:
    """
    Compute sorted_token_indices via Triton stable argsort kernel.
    Returns a torch int32 tensor of shape (M,) with permutation [0..M-1] sorted by flat values (stable).
    """
    device = flat.device
    M = flat.numel()
    NUM_EXPERTS = 256
    out = torch.empty(M, dtype=torch.int32, device=device)
    # base_excl and tie_rank buffers
    base_excl = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
    tie_rank = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)

    grid = (M,)
    stable_argsort_kernel[grid](
        flat, out, base_excl, tie_rank, M,
        NUM_EXPERTS=NUM_EXPERTS, BLOCK_SIZE=1024
    )
    return out


def _compute_expert_offsets_triton(flat: torch.Tensor) -> torch.Tensor:
    """
    Compute expert_offsets (length num_experts+1) via Triton:
    offsets[:num_experts] = inclusive prefix sums of counts of each expert id in flat
    offsets[num_experts] = total_count + 1
    """
    device = flat.device
    M = flat.numel()
    NUM_EXPERTS = 256

    # Per-key counts (initialized to zeros)
    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
    # Total count
    total = torch.zeros(1, dtype=torch.int32, device=device)

    # Triton atomic count kernel
    _count_per_key_atomic_kernel[(1,)](
        flat, counts, total, M,
        NUM_EXPERTS=NUM_EXPERTS, BLOCK_SIZE=1024
    )

    # Inclusive scan via Triton
    incl_scan = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
    _inclusive_scan_counts_kernel[(1,)](counts, incl_scan, NUM_EXPERTS=NUM_EXPERTS)

    # Construct offsets tensor
    offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    offsets[:NUM_EXPERTS] = incl_scan
    offsets[NUM_EXPERTS] = total[0] + 1
    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: int32 tensor of shape (batch_size, seq_len, num_experts_per_tok), values in [0, 255]
        Returns:
          sorted_token_indices: int32 tensor (num_tokens,) sorted by flattened values (stable)
          expert_offsets: int32 tensor (num_experts+1,) inclusive counts per expert + 1
        """
        flat = topk_idx.reshape(-1)  # 1D int32 tensor on device

        # Compute sorted_token_indices using Triton stable argsort (TRITON-ONLY)
        sorted_token_indices = _run_triton_sorted_indices(flat)

        # Compute expert_offsets using Triton for counts and scan (TRITON-ONLY)
        expert_offsets = _compute_expert_offsets_triton(flat)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
