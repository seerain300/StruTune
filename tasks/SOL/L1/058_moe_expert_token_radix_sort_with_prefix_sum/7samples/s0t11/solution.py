import torch
import triton
import triton.language as tl


@triton.jit
def triton_bincount_flat(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Triton kernel to compute per-element bin counts into counts_ptr[0:256].
    Assumes x_ptr contains int32 values in [0, 255]. Each program processes BLOCK elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load flattened values
    # If x_ptr is int64, tl.load will handle it; here inputs are int32 from get_inputs.
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Increment counts for each element in [0, 255]
    # Note: Triton will vectorize these atomic adds; this is safe under mask.
    for i in range(0, 256):
        # Create a boolean mask for x == i; since x is int32, comparison is fine.
        is_i = x == i
        # Only increment for valid offsets
        tl.atomic_add(counts_ptr + i, tl.where(mask & is_i, 1, 0))


def triton_bincount(topk_idx_flat: torch.Tensor):
    """
    Compute per-expert counts using Triton. Returns int32 tensor of length 256 on the same device.
    Assumes topk_idx_flat is int32 and contains values in [0, 255].
    """
    N = topk_idx_flat.numel()
    # Ensure int32 for Triton
    x = topk_idx_flat.to(torch.int32).contiguous()
    counts = torch.zeros(256, dtype=torch.int32, device=x.device)

    # Launch kernel
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    triton_bincount_flat[grid](x, counts, N, BLOCK=BLOCK)

    return counts


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - sorted_token_indices: stable argsort of flattened indices, int32
        - expert_offsets: inclusive prefix sum of bincounts over 256 experts, int64 (257,)
        """
        # Flatten topk_idx
        flat = topk_idx.reshape(-1)

        # sorted_token_indices must match original dtype (int32) and behavior
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # Triton bincount for per-expert counts
        counts = triton_bincount(flat)  # int32, length 256

        # Compute inclusive prefix sum using torch (safe and fast for small vectors)
        # This matches the original torch.bincount(...).cumsum(0) behavior.
        prefix = torch.cumsum(counts.to(torch.int64), dim=0)  # int64
        expert_offsets = torch.zeros(257, dtype=torch.int64, device=flat.device)
        expert_offsets[1:] = prefix  # offset[0] = 0

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
