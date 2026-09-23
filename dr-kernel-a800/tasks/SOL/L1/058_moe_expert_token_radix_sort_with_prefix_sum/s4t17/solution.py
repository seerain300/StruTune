import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: block-wise reduction histogram into counts
# counts[expert_id] = number of occurrences of expert_id in flat
if TRITON_AVAILABLE:
    @triton.jit
    def _histogram_block_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        start = pid * BLOCK
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N

        # Load a block of indices (int32), masked lanes get 0
        idx = tl.load(flat_ptr + offsets, mask=mask, other=0)

        # Vector of counts for each expert bin (length num_experts)
        # For each lane, increment the corresponding bin if mask is true
        # We do this by iterating over bin ids and adding 1 where idx == bin_id.
        # Note: idx and counts_ptr are int32, operations are vectorized.
        for i in range(num_experts):
            # mask_i is true for lanes where idx == i
            mask_i = mask & (idx == i)
            # Convert boolean mask to int32 count
            cnt_i = mask_i.to(tl.int32)
            # Accumulate per bin across lanes
            tl.atomic_add(counts_ptr + i, tl.sum(cnt_i, axis=0))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]

        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Triton counts per expert id (int32), num_experts=256
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)

        # Kernel launch configuration: process BLOCK elements per program
        BLOCK = 2048  # larger block to reduce grid size; adjust warps accordingly
        grid = (triton.cdiv(N, BLOCK),)

        if TRITON_AVAILABLE:
            _histogram_block_kernel[grid](
                flat, counts, N,
                num_experts=256, BLOCK=BLOCK, num_warps=8
            )
        else:
            # Fallback: if Triton is not available, use torch.bincount
            counts = torch.bincount(flat.to(torch.int64), minlength=256).to(torch.int32)

        # Compute expert offsets: inclusive prefix sum starting from 0
        # Shape: (num_experts + 1,) = (257,)
        expert_offsets = torch.zeros(257, dtype=torch.int32, device=flat.device)
        running = 0
        for i in range(256):
            running += counts[i]
            expert_offsets[i + 1] = running

        # Stable sort of flattened indices (values-only, independent of num_experts)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
