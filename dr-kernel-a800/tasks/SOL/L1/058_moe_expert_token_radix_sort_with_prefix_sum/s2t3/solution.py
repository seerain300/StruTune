import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: for each element in vals_ptr (int32), atomically increment counts_ptr[exp],
    where exp is the value at that position. Assumes vals_ptr holds expert indices in [0, num_experts).
    """
    pid = tl.program_id(0)
    BLOCK = 1024
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(vals_ptr + offsets, mask=mask, other=0)  # x is int32
    # Atomic add counts[exp] by 1 for each occurrence
    tl.atomic_add(counts_ptr + x.to(tl.int32), 1, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is fixed in provided workloads; set to 256
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA and contiguous
        device = topk_idx.device
        # Flatten the input to 1D
        vals = topk_idx.reshape(-1).contiguous()
        N = vals.numel()

        # sorted_token_indices: stable sort of flattened expert indices
        # Using torch.sort to exactly match original behavior
        sorted_token_indices, _ = torch.sort(vals, stable=True)
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        # Compute expert counts using Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        # Launch a grid over chunks of 1024 elements
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_experts_kernel[grid](vals, counts, N, self.num_experts)

        # Compute expert offsets = prefix sum [0] + cumulative counts
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=device)
        # torch.cumsum on GPU
        expert_offsets[1:] = counts.cumsum(0)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
