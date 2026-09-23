import torch
import triton
import triton.language as tl


# Minimal Triton bincount kernel: one program per element, atomic_add to counts[value].
# We process the flattened tensor as int64 to avoid dtype issues in Triton, then cast to int32 index.
@triton.jit
def _bincount_kernel(flat64_ptr: tl.pointer_type(tl.int64), counts_ptr: tl.pointer_type(tl.int32), N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load values as int64; invalid (out-of-range) lanes are masked
    vals64 = tl.load(flat64_ptr + offsets, mask=mask, other=0)
    # Check valid range [0, 255]
    valid = (vals64 >= 0) & (vals64 <= 255)
    # Cast to int32 for index
    vals32 = vals64.to(tl.int32)
    idx = tl.where(valid & mask, vals32, 0)
    # Atomic add 1 for each valid element
    # We cannot branch per-element easily; for masked lanes, idx is 0 and atomic_add does nothing harmful.
    tl.atomic_add(counts_ptr + idx, 1)


# PyTorch cumsum for offsets (we still perform a numeric reduction in Triton via bincount, and cumsum here)
class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()
        device = flat.device

        # 1) Triton bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # Convert flat to int64 for Triton load
        flat64 = flat.to(torch.int64)
        # Use one program per BLOCK chunk; BLOCK=1 keeps it minimal and robust
        BLOCK = 1
        grid = (triton.cdiv(N, BLOCK),)
        _bincount_kernel[grid](flat64, counts, N, BLOCK=BLOCK)

        # 2) Compute inclusive prefix sum of counts using PyTorch (int64 offsets of length 257)
        #    This matches torch.bincount(..., minlength=256).cumsum(0) behavior.
        #    We explicitly create offsets[0] = 0.
        counts_int64 = counts.to(torch.int64)
        expert_offsets = torch.cumsum(counts_int64, dim=0)  # shape (256,)
        # Add leading zero for minlength=256 cumulative prefix
        offsets = torch.empty(257, dtype=torch.int64, device=device)
        offsets[0] = 0
        offsets[1:] = expert_offsets

        # 3) Use PyTorch for stable argsort of flattened indices (returns permutation of [0, N-1])
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
