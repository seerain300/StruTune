import torch
import triton
import triton.language as tl


# Triton kernel: perform stable selection sort to produce sorted_token_indices (permutation).
# We keep two arrays: values[pos] and indices[pos], initially values[pos] = flattened indices,
# indices[pos] = original positions [0..N-1]. We repeatedly find the minimum among remaining
# elements, for ties (equal values), choose the smallest original index (stable).
@triton.jit
def _selection_sort_stable(values_ptr, indices_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # One program per output position 'pos' (pos = 0..N-1)
    pos = tl.program_id(0)

    # Initialize min_val and min_idx
    # Assume int32 values and indices
    min_val = tl.load(values_ptr + pos)
    min_idx = pos

    # Scan remaining elements to find the minimum; for ties, choose smallest index
    # We iterate i from pos+1 to N-1 in steps of BLOCK. For each block, we process element-by-element.
    # Note: We only need one scalar min per pos, but Triton requires block-wise loops for arbitrary N.
    # We do a simple per-element update inside the loop over i.
    for i in range(pos + 1, N):
        val_i = tl.load(values_ptr + i)
        idx_i = i
        # If val_i < min_val or (val_i == min_val and idx_i < min_idx): update
        update = (val_i < min_val) | ((val_i == min_val) & (idx_i < min_idx))
        # Update min_val and min_idx where needed
        min_val = tl.where(update, val_i, min_val)
        min_idx = tl.where(update, idx_i, min_idx)

    # Now swap values[pos] with min_val and indices[pos] with min_idx
    # Read current pos values
    val_pos = tl.load(values_ptr + pos)
    idx_pos = tl.load(indices_ptr + pos)
    # Perform the swap
    tl.store(values_ptr + pos, min_val)
    tl.store(indices_ptr + pos, min_idx)
    tl.store(values_ptr + min_idx, val_pos)
    tl.store(indices_ptr + min_idx, idx_pos)


# Triton kernel: histogram via atomic adds. Each program processes BLOCK elements.
@triton.jit
def _histogram_atomic(values_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load values; use 0 for masked lanes to avoid spurious adds
    vals = tl.load(values_ptr + offs, mask=mask, other=0)
    # Each lane atomically adds 1 to counts[vals]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton kernel: inclusive prefix sum over counts (length M=256) into offsets (length M+1).
@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    # Single program instance performs scan over M entries
    acc = tl.zeros((), dtype=tl.int32)
    acc = tl.load(offsets_ptr + 0)  # initialize acc from offsets[0]
    for i in range(0, M):
        ci = tl.load(counts_ptr + i)
        acc += ci
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensors
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Stable sort via Triton selection sort: produce permutation indices
        # Prepare working copies
        values = flat.clone()
        # indices buffer initialized to original positions [0..N-1]
        indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        # Launch one program per position
        grid = (N,)
        _selection_sort_stable[grid](values, indices, N, BLOCK=1)

        # 2) Histogram via Triton atomic adds
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_atomic[grid_hist](values, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over counts
        offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return indices, offsets