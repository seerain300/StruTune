import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values from flat; out-of-range lanes get 0 (ignored by mask)
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Accumulate atomics only for valid lanes
    # For each element, atomically add 1 into counts[vals[i]]
    for i in range(BLOCK):
        idx = int(vals[i])  # flat is int32; Triton will handle casting
        # Only do atomic if this lane was valid
        if mask[i]:
            tl.atomic_add(counts_ptr + idx, 1)


@triton.jit
def inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    # Single program computes the inclusive prefix sum of counts into offsets
    running = tl.zeros((), dtype=tl.int64)
    offsets_ptr[0] = 0  # inclusive starting at 0
    for k in range(1, L):
        running += tl.load(counts_ptr + k - 1)  # int32 -> implicit cast to int64 on store
        offsets_ptr[k] = running


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure data is on the same device; get_inputs returns device-aware tensors
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton bincount: counts of each expert id in [0, 255]
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # tune as needed; 1024 is a good default
        grid = (triton.cdiv(N, BLOCK),)
        bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_kernel[(1,)](counts, offsets, L=257)

        # 3) Use PyTorch for stable argsort of flattened indices: permutation of [0, N-1]
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets