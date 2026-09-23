import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: count per expert id for flat
if TRITON_AVAILABLE:
    @triton.jit
    def _count_per_key_kernel(flat_ptr, counts_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
        BLOCK_SIZE = 1024
        i = 0
        while i < M:
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < M
            vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
            # For each valid element in the chunk, atomic add to counts
            for k in range(BLOCK_SIZE):
                if mask[k]:
                    val = vals[k]
                    # Only count within valid expert range
                    if (val >= 0) & (val < NUM_EXPERTS):
                        tl.atomic_add(counts_ptr + val, 1)
            i += BLOCK_SIZE

    # Triton kernel: inclusive scan of counts -> incl_scan[e] = sum(counts[0..e])
    @triton.jit
    def _scan_inclusive_kernel(counts_ptr, incl_ptr, NUM_EXPERTS: tl.constexpr):
        total = 0
        for e in range(NUM_EXPERTS):
            total += tl.load(counts_ptr + e)
            tl.store(incl_ptr + e, total)

    # Triton kernel: stable argsort via ranks. For each j, rank_j = base_excl[flat[j]-1] (if >0) + tie_breaker.
    # We update counts[k] only after writing out, so base_excl for other keys remains correct.
    @triton.jit
    def _stable_argsort_rank_kernel(flat_ptr, out_ptr, counts_ptr, incl_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        # Process flat in chunks; for each element j, compute rank and store to out[j]
        i = 0
        while i < M:
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < M
            vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32

            for k in range(BLOCK_SIZE):
                if mask[k]:
                    j = i + k
                    val = vals[k]  # key in [0, 255]
                    # base_excl: number of elements strictly less than val
                    base_prev = 0
                    if val > 0:
                        base_prev = tl.load(incl_ptr + (val - 1))
                    # tie_breaker: among elements with same key, earlier positions come first
                    # We can't access original indices here, but we can use position j as tie-breaker:
                    # For equal keys, count previous positions t < j with same key to ensure stable ordering.
                    tie = 0
                    # Scan previous positions in this chunk to count ties
                    for t in range(k):
                        if mask[t]:
                            if vals[t] == val and (i + t < j):
                                tie += 1
                    rank = base_prev + tie
                    tl.store(out_ptr + j, rank)

                    # Update counts only for this key (for keys other than val, counts remain unchanged).
                    if (val >= 0) & (val < NUM_EXPERTS):
                        tl.atomic_add(counts_ptr + val, 1)

            i += BLOCK_SIZE

    # Triton kernel: produce expert_offsets = inclusive scan + 1 at the end (total + 1)
    @triton.jit
    def _finalize_offsets_kernel(counts_ptr, incl_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
        # Compute total = sum(counts)
        total = 0
        for e in range(NUM_EXPERTS):
            total += tl.load(counts_ptr + e)
        # Copy incl into offsets[0:NUM_EXPERTS]
        for e in range(NUM_EXPERTS):
            tl.store(offsets_ptr + e, tl.load(incl_ptr + e))
        # Set last element to total + 1
        tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


def _compute_expert_offsets_triton(flat: torch.Tensor) -> torch.Tensor:
    """
    Compute expert_offsets = inclusive counts per expert id + 1 at the end.
    Uses Triton for counting and scan. Returns int32 tensor of length (num_experts+1).
    """
    assert flat.is_cuda, "Triton requires CUDA tensors"
    NUM_EXPERTS = 256
    M = flat.numel()

    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
    _count_per_key_kernel[(1,)](flat, counts, M, NUM_EXPERTS=NUM_EXPERTS)

    incl_scan = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
    _scan_inclusive_kernel[(1,)](counts, incl_scan, NUM_EXPERTS=NUM_EXPERTS)

    offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
    _finalize_offsets_kernel[(1,)](counts, incl_scan, offsets, NUM_EXPERTS=NUM_EXPERTS)
    return offsets


def _run_triton_sorted_indices(flat: torch.Tensor) -> torch.Tensor:
    """
    Triton-based stable argsort to produce sorted_token_indices (stable=True) for flat.
    This kernel computes ranks using per-key base_excl from counts+scan and adds a tie-breaker
    based on position j for equal keys. Returns int32 tensor of length M.
    """
    assert flat.is_cuda, "Triton requires CUDA tensors"
    NUM_EXPERTS = 256
    M = flat.numel()
    out = torch.empty(M, dtype=torch.int32, device=flat.device)

    # First, compute counts and incl_scan
    counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
    incl_scan = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=flat.device)

    _count_per_key_kernel[(1,)](flat, counts, M, NUM_EXPERTS=NUM_EXPERTS)
    _scan_inclusive_kernel[(1,)](counts, incl_scan, NUM_EXPERTS=NUM_EXPERTS)

    # Assign ranks using incl_scan. We must ensure counts are updated only after writing out
    # so base_excl for other keys remains correct when computing later j's.
    _stable_argsort_rank_kernel[(1,)](flat, out, counts, incl_scan, M, NUM_EXPERTS=NUM_EXPERTS, BLOCK_SIZE=1024)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor: topk_idx
        topk_idx = args[0]
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        # Flatten
        flat = topk_idx.reshape(-1)
        # Ensure device is CUDA for Triton
        if not flat.is_cuda:
            flat = flat.to('cuda')
        # Compute sorted_token_indices using Triton argsort (stable)
        sorted_token_indices = _run_triton_sorted_indices(flat)
        # Compute expert_offsets using Triton
        expert_offsets = _compute_expert_offsets_triton(flat)
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
