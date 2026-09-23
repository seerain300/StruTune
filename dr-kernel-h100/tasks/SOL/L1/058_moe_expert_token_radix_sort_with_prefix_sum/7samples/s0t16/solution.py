import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: bincount for int32 inputs into 256 bins.
# flat: pointer to int32, length N
# counts: pointer to int32, length 256 (will be zero-initialized on host)
# Grid: one program per BLOCK elements; lanes iterate over 256 bins.
if TRITON_AVAILABLE:
    @triton.jit
    def bincount_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # Accumulate per bin: for i in 0..255, if vals==i and in range, atomic add 1
        for i in range(256):
            is_i = vals == i
            # Only increment if in range and mask is true
            tl.atomic_add(counts_ptr + i, tl.where(mask & is_i, 1, 0))

    # Triton kernel: inclusive prefix sum over a 1D int32 vector of length L, write int32 result
    @triton.jit
    def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.int32):
        # Single program does a simple scan. L is expected to be 257 (num_experts+1).
        total = 0
        for i in range(L):
            vi = tl.load(x_ptr + i)  # int32
            total += vi
            tl.store(y_ptr + i, total)  # int32 result


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)

        # 1) Compute per-expert counts with Triton (int32), length 256
        N = flat.numel()
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        if TRITON_AVAILABLE:
            # Choose a reasonable block size; 1024 works well for typical N
            BLOCK = 1024
            grid = (triton.cdiv(N, BLOCK),)
            bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)
        else:
            # Fallback: use torch.bincount for correctness if Triton is not available
            counts = torch.bincount(flat, minlength=256)

        # 2) Compute inclusive prefix sum of counts in Triton (int32), then convert to int64 to match original
        offsets_i32 = torch.empty(257, dtype=torch.int32, device=flat.device)
        if TRITON_AVAILABLE:
            inclusive_prefix_sum_kernel[(1,)](counts, offsets_i32, L=257)
        else:
            # Fallback: torch.cumsum for correctness
            offsets_i32 = counts.cumsum(0)

        expert_offsets = offsets_i32.to(torch.int64)  # match original dtype (int64)

        # 3) sorted_token_indices: stable argsort of flattened indices, return int32 permutation
        # This matches the original run exactly.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
