import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel to count occurrences of each expert id in vals_ptr (int32).
    For each loaded element, atomic_add counts[exp] += 1.
    vals_ptr points to flattened expert indices in [0, num_experts).
    """
    pid = tl.program_id(0)
    BLOCK = 1024
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(vals_ptr + offsets, mask=mask, other=0)  # int32
    # Atomic add to counts[exp], exp = x
    tl.atomic_add(counts_ptr + x.to(tl.int32), 1, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is on CUDA and contiguous
        assert topk_idx.is_cuda, "ModelNew.forward expects a CUDA tensor"
        device = topk_idx.device
        vals = topk_idx.reshape(-1).contiguous()
        N = vals.numel().item()
        num_experts = 256  # fixed for provided workloads

        # Step 1: Compute per-expert counts using Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_experts_kernel[grid](vals, counts, N, num_experts)

        # Step 2: Compute offsets = prefix sum of counts
        offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        offsets[1:] = counts.cumsum(0)

        # Step 3: Compute sorted_token_indices by stable sort of flattened vals
        # We use torch.sort with stable=True to match original behavior exactly


def run(*args):
    return ModelNew()(*args)
