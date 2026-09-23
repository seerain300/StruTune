import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(
    flat_ptr,          # *int32, 1D input flattened tokens
    counts_ptr,        # *int32, 1D output histogram length=num_experts
    n_elements,        # int32, total number of elements in flat
    num_experts: tl.constexpr,  # number of experts (compile-time constant for kernel)
    BLOCK_SIZE: tl.constexpr,   # block size for vectorized processing
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load a chunk of flat; other=0 for out-of-range
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # For each value in the chunk, atomically add 1 to counts[val]
    # Note: vals are int32 indices in [0, num_experts-1]
    for i in range(BLOCK_SIZE):
        val = vals[i]
        if mask[i]:
            # bounds check is implicit by num_experts; Triton will handle int32 indexing
            tl.atomic_add(counts_ptr + val, 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure we have a valid device; get_inputs provides CUDA tensors
        device = topk_idx.device
        # Flatten to 1D and ensure int32
        flat = topk_idx.reshape(-1).to(torch.int32)
        n = flat.numel()
        num_experts = 256  # per provided setup; treat as constant

        # Allocate counts on device
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)

        # Launch Triton histogram kernel
        # Use a moderate block size to vectorize; 1024 works well for typical sizes
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](
            flat,
            counts,
            n,
            num_experts=num_experts,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        # Compute expert_offsets via torch.cumsum on counts (inclusive)
        # We need length = num_experts + 1
        offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        # inclusive cumsum: offsets[1:] = cumsum(counts), offset[0] = 0
        offsets[1:] = torch.cumsum(counts, dim=0)

        # sorted_token_indices is the stable sort indices of flat.
        # Keep torch.sort for correctness and simplicity; cast to int32 to match original.
        sorted_token_indices = torch.sort(flat, stable=True)[1].to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
