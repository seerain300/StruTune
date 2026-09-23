import torch
import triton
import triton.language as tl


@triton.jit
def _bincount_and_prefix_offsets_kernel(flat64_ptr: tl.pointer_type(tl.int64),
                                         counts_ptr: tl.pointer_type(tl.int32),
                                         offsets_ptr: tl.pointer_type(tl.int64),
                                         N: tl.constexpr):
    # We run a single program and loop over the entire flat array.
    # For each element, if it is in [0, 255], increment counts[element].
    # Then compute inclusive prefix sums and write to offsets[0..256], and to full 257-element buffer.

    # Initialize counts to zeros (assumed by caller)
    # Note: counts_ptr points to a 256-length array (int32).
    # offsets_ptr points to a 257-length array (int64); we write offsets[0] = 0 and offsets[1..] = prefix sums.

    # Loop over all elements; N is the length of the flattened tensor.
    total = tl.zeros((), dtype=tl.int64)
    for i in range(N):
        # Load value as int64
        v = tl.load(flat64_ptr + i)
        # Check valid range
        is_valid = (v >= 0) & (v < 256)
        # Convert to int32 index for atomic add
        idx = v.to(tl.int32)
        # Atomic add 1 to counts[idx] if valid
        # This avoids building large per-block arrays and reduces complexity.
        # Triton atomic_add supports int32.
        tl.atomic_add(counts_ptr + idx, 1, mask=is_valid)

    # Compute inclusive prefix sum for counts and write to offsets[1..256]
    # offsets[0] should be 0. We will write the full 257-length buffer by manually setting j=0 and j in [1..256].
    # We use j as a loop counter; Triton supports range loops with compile-time constants.

    # First, set offsets[0] = 0
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int64))

    # Then, for j in 1..256, compute prefix and store
    for j in range(256):
        # Read counts[j] and accumulate
        c = tl.load(counts_ptr + j)
        total += c.to(tl.int64)
        tl.store(offsets_ptr + 1 + j, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # 1) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        #    This matches the original run's behavior for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # 2) Triton kernel to compute bincount counts (int32) and offsets (int64, length 257)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        offsets = torch.empty(257, dtype=torch.int64, device=device)
        # Convert flat to int64 for Triton load
        flat64 = flat.to(torch.int64)

        # Launch single-program kernel
        _bincount_and_prefix_offsets_kernel[(1,)](flat64, counts, offsets, N)

        return sorted_token_indices, offsets