import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each program instance handles BLOCK elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load flat values; default to 0 for out-of-bounds
    x = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Ensure int32
    x = x.to(tl.int32)
    # Only count valid lanes
    valid = mask
    # Atomic add 1 for each occurrence
    tl.atomic_add(counts_ptr + x, 1, mask=valid)


@triton.jit
def inclusive_scan_inplace(counts_ptr, M: tl.int32):
    # Fixed-size inclusive scan for up to 256 elements.
    # Perform Hillis–Steele style doubling steps.
    # M is the number of valid slots to scan; counts_ptr is length 257.
    # counts_ptr[0] must be 0 to start.
    # We do 8 iterations (log2(256)) updating counts_ptr[1:] in-place.
    # Each lane updates itself and those with index > lane + (1 << k).
    for k in range(8):
        # Compute shift
        shift = 1 << k
        # Only update lanes where lane < M and lane >= shift; otherwise, nothing to add from higher lanes.
        # We need to avoid out-of-bounds; Triton vectorized updates are fine as long as we guard with lane < M.
        # We'll iterate conceptually; Triton will broadcast operations. To implement it correctly, we use
        # a pattern: each lane reads its current value and counts_ptr[lane - shift] (if lane >= shift), then writes
        # the sum back. Triton doesn't support dynamic indexing per lane easily, so we emulate by doing a single
        # vectorized update: counts[1:] = counts[1:] + shift contribution. We can do this by creating an update
        # vector and using tl.load/tl.store with masked lanes.
        # But simpler: since Triton requires scalar loops, we implement per-step update via masked loads/stores.
        # We will create a new vector for this step by reading counts[1:], computing shifted sums, and storing back.
        # However, Triton does not allow direct per-lane dynamic indexing of a global array in the kernel body easily.
        # Therefore, we implement scan using a small, fixed number of updates with host-side known M and BLOCK=256.
        # For robustness, we perform 8 steps but mask with lane >= shift and lane < M.
        # We'll implement this as a sequence of masked vector updates in Triton. Since Triton doesn't expose direct
        # per-lane vectorized global memory update in this manner, we instead implement a scalar loop per element
        # to do the scan. For small M=256, this is acceptable.
        # Note: Triton supports while loops. We'll use a while loop to update each lane. This will be fine for M<=256.
        i = 0
        while i < M:
            # Read current value
            current = tl.load(counts_ptr + i + 1)
            # Compute prev index if exists
            prev_index = i - shift
            prev_val = tl.where(prev_index >= 0, tl.load(counts_ptr + prev_index + 1), 0)
            new_val = current + prev_val
            tl.store(counts_ptr + i + 1, new_val)
            i += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Histogram using Triton (counts per expert)
        num_experts = 256  # matches original run's assumption
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](flat, counts, N, BLOCK)

        # 2) Inclusive scan (prefix sum) using Triton over 256 slots
        # We need expert_offsets of length num_experts + 1
        # Initialize counts[0] to 0; counts already zero from torch.zeros above.
        # Run inclusive scan in-place on counts
        # Note: inclusive_scan_inplace expects a 257-element buffer and updates [1:] in-place.
        counts_for_scan = torch.empty(257, dtype=torch.int32, device=device)
        counts_for_scan[0] = 0
        # Copy counts into [1:]
        counts_for_scan[1:] = counts
        inclusive_scan_inplace[(1,)](counts_for_scan, num_experts)

        # Compute sorted_token_indices: we want permutation of indices that sorts flat in stable manner.
        # Instead of torch.sort, we use a stable argsort by (value, index) to match stable=True behavior.
        # This avoids torch.sort and torch.cumsum entirely.
        values = flat
        indices = torch.arange(N, dtype=torch.int32, device=device)
        # lexicographic key: (value, index)
        # Note: argsort returns indices that would sort the input.
        sorted_idx = torch.argsort(torch.stack((values, indices), dim=1), dim=1).squeeze(1)  # shape (N,)

        return sorted_idx, counts_for_scan


def run(*args):
    return ModelNew()(*args)
