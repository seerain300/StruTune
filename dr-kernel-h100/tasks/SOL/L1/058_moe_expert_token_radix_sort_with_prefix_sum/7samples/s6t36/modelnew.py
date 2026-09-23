import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    # Single-program loop over N to build histogram via atomic adds
    # flat_ptr: *int32, length N
    # counts_ptr: *int32, length NUM_CLASSES
    for i in range(0, N):
        v = tl.load(flat_ptr + i)
        tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, scan_ptr, NUM_CLASSES: tl.int32):
    # Single-program inclusive scan over NUM_CLASSES
    # counts_ptr: *int32, length NUM_CLASSES
    # scan_ptr: *int32, length (NUM_CLASSES + 1)
    # Initialize scan[0] = 0
    tl.store(scan_ptr + 0, 0)
    total = 0
    for c in range(0, NUM_CLASSES):
        total += tl.load(counts_ptr + c)
        tl.store(scan_ptr + (c + 1), total)


@triton.jit
def _global_argsort_stable_kernel(flat_ptr, out_idx_ptr, scan_ptr, N: tl.int32, NUM_CLASSES: tl.int32):
    # Fill out_idx stably using the inclusive scan ranges.
    # out_idx_ptr: *int32, length N (initialized to zeros).
    # scan_ptr: *int32, length (NUM_CLASSES + 1).
    for c in range(0, NUM_CLASSES):
        start = tl.load(scan_ptr + c)  # inclusive
        end = tl.load(scan_ptr + (c + 1))  # exclusive
        # For each original index i with flat[i] == c, place i at out_idx[start + j].
        # We emulate stable insertion by processing i in increasing order (Python loop).
        # In Triton, this requires a loop over N. We rely on per-class equality checks.
        # Note: This is O(N) per class but NUM_CLASSES=256, acceptable for given N.
        for i in range(0, N):
            v = tl.load(flat_ptr + i)
            if v == c:
                # Find number of tokens already placed for class c: start is the next available.
                tl.store(out_idx_ptr + start, i)
                start += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch_size, seq_len, num_experts_per_tok)
        assert topk_idx.dim() == 3, "topk_idx must be 3D"
        # Flatten to 1D for global sort
        flat = topk_idx.reshape(-1)
        # Ensure int32 for Triton kernels
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)
        flat = flat.contiguous()

        N = flat.numel()
        NUM_CLASSES = 256  # same as original code

        # 1) Compute counts per class using Triton
        counts = torch.zeros(NUM_CLASSES, dtype=torch.int32, device=flat.device)
        _hist_kernel[(1,)](flat, counts, N, NUM_CLASSES)

        # 2) Compute inclusive scan of counts using Triton
        scan = torch.empty(NUM_CLASSES + 1, dtype=torch.int32, device=flat.device)
        _inclusive_scan_kernel[(1,)](counts, scan, NUM_CLASSES)

        # 3) Compute sorted_token_indices stably using Triton
        out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
        _global_argsort_stable_kernel[(1,)](flat, out_idx, scan, N, NUM_CLASSES)

        # 4) Build expert_offsets: original code sets offsets[1:] = cumulative counts,
        #    but we don't have original flat here; we use scan[1:] which is inclusive cumulative of original flat.
        #    Since the original uses counts derived from original flat, and we computed counts here from flat,
        #    scan[1:] already represents offsets of original flat. Thus, expert_offsets = scan[1:].
        expert_offsets = scan[1:]

        # Return with exact shapes/dtypes as original:
        # sorted_token_indices: shape (N,), dtype int32
        # expert_offsets: shape (num_experts + 1,), dtype int32
        return out_idx, expert_offsets