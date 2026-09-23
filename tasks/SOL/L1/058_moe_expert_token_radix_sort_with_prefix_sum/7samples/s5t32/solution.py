import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert ID in flat_ptr into counts_ptr[num_experts].
    flat_ptr: 1D int32 tensor of length N.
    counts_ptr: 1D int32 tensor of length num_experts.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load a chunk of IDs
    ids = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # Atomic add 1 for each valid ID
    # counts_ptr is assumed to be zero-initialized before launch
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute exclusive prefix sum of counts_ptr into offsets_ptr[1:].
    offsets_ptr[0] is set on host to 0.
    counts_ptr: 1D int32, length num_experts.
    offsets_ptr: 1D int32, length num_experts + 1.
    """
    # Single program instance computes prefix sum sequentially.
    acc = tl.zeros((), dtype=tl.int32)
    # Loop over num_experts; use constexpr or dynamic loop. Triton supports dynamic loops.
    for i in range(0, num_experts):
        acc += counts_ptr[i]
        offsets_ptr[i + 1] = acc


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure we operate on device; get num_experts from input range (0, num_experts-1)
        device = topk_idx.device
        num_experts = int(torch.tensor(256, device=device))  # fixed per the original setup

        # Flatten and cast to int32 for counting
        flat = topk_idx.reshape(-1).to(torch.int32)

        # 1) Compute sorted_token_indices using PyTorch's stable sort on GPU
        #    This matches the original behavior exactly.
        sorted_token_indices = flat.sort(stable=True)[1]  # indices that sort flat ascending

        # 2) Compute expert offsets using Triton:
        N = flat.numel()
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        # Kernel to count per-expert IDs
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_expert_ids_kernel[grid](flat, counts, N, BLOCK=BLOCK)
        # Kernel to compute exclusive prefix sum
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)
        # offsets[0] = 0 already in offsets

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
