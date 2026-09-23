import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Create per-lane values by gathering flat[offs]
    # Note: flat_ptr is int32*, we load as int32.
    idx = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32

    # For each possible expert id 0..255, atomic add 1 for lanes where idx == i.
    # This is simple and avoids dynamic indexing complexity.
    for i in range(256):
        # Only count valid lanes
        eq = mask & (idx == i)
        # Atomic add 1 for those equal lanes. counts_ptr is int32*.
        tl.atomic_add(counts_ptr + i, 1, mask=eq)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Single-program inclusive prefix sum over counts (int32) into offsets (int64).
    running = tl.zeros((), dtype=tl.int64)  # int64 running sum
    # i = 0 already has running = 0; offsets[0] can be set on host.
    for i in range(1, L):
        val_i = tl.load(counts_ptr + i)  # int32
        val_i64 = val_i.to(tl.int64)
        running += val_i64
        tl.store(offsets_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure we operate on CUDA tensors; Triton kernels require CUDA.
        if not topk_idx.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors. Move inputs to .cuda().")

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Triton bincount of flattened indices into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # tune as needed; 1024 is a safe default
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        #    The original returns int32 for this output.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets