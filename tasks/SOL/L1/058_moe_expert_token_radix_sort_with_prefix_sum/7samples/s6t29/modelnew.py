import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Triton histogram of flat values (int32) into counts_ptr (length NUM_CLASSES).
    Each program handles one class and scans the flat vector to count occurrences.
    """
    cls = tl.program_id(0)
    if cls < NUM_CLASSES:
        total = tl.zeros((), dtype=tl.int32)
        # Loop over flat in chunks of size BLOCK to count occurrences of cls
        BLOCK = 1024
        for i in range(0, N, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < N
            vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
            # Count occurrences of cls in this chunk
            # Note: mask ensures we only count valid offsets
            total += tl.sum((vals == cls) & mask, axis=0)
        tl.store(counts_ptr + cls, total)


@triton.jit
def _inclusive_scan_inplace(inclusive_ptr, M: tl.constexpr):
    """
    In-place sequential inclusive scan on a vector of length M.
    inclusive_ptr[0..M-1] is assumed to be initialized with per-class counts.
    The kernel writes inclusive_ptr[i] = sum(inclusive_ptr[0..i]).
    """
    # Each program handles one element; we launch a 1D grid with M programs.
    idx = tl.program_id(0)
    if idx < M:
        # Load current count
        current = tl.load(inclusive_ptr + idx)
        # Compute inclusive sum by sequential accumulation
        # We'll use a loop over k in [0, M) to accumulate; for small M (256), this is fine.
        inclusive_val = current
        for k in range(idx):
            inclusive_val += tl.load(inclusive_ptr + k)
        tl.store(inclusive_ptr + idx, inclusive_val)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward that:
        - Uses torch.argsort(stable=True) to compute sorted_token_indices of shape (N,), dtype int32.
        - Uses Triton kernels to compute expert_offsets from original topk_idx:
          - Triton histogram to count occurrences per expert.
          - Triton inclusive scan to produce cumulative counts.
        Returns (sorted_token_indices, expert_offsets).
        """
        # Ensure tensor on CUDA for Triton kernels
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda", non_blocking=True)

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        # torch.argsort for correctness
        # Note: torch.argsort returns indices (stable), which is exactly what we need.
        # The original function sorts flat; values are not used after that, so this is fine.
        # Ensure int32 for offsets and count comparisons; argsort indices are already int64 by default,
        # but sorted_token_indices is expected as int32. We can return int64 if needed, but original
        # code returns int32. torch.argsort returns long (int64), we can cast to int32 for consistency.
        sorted_token_indices = torch.argsort(flat, stable=True).to(torch.int32)

        # Compute expert offsets from original flat values (original behavior)
        num_experts = 256  # same as in the original code setup
        # Triton histogram
        counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        grid_hist = (num_experts,)
        _hist_kernel[grid_hist](flat, counts, flat.numel(), NUM_CLASSES=num_experts)

        # Triton inclusive scan to get inclusive cumulative counts
        inclusive = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        # We can initialize inclusive with counts and run the scan kernel
        inclusive.copy_(counts)
        _inclusive_scan_inplace[(num_experts,)](inclusive, M=num_experts)

        # Construct expert_offsets as in the original: offsets[0]=0, offsets[1:]=inclusive
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = inclusive

        return sorted_token_indices, expert_offsets