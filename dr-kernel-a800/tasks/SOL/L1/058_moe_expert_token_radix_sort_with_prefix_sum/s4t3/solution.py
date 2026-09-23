import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_block_atomic_kernel(
    inp_ptr,          # *int32, flattened input topk_idx
    N,                # int32, number of elements
    counts_ptr,       # *int32, size=NUM_EXPERTS, output per-expert counts
    NUM_EXPERTS: tl.constexpr,  # number of experts (256)
    BLOCK: tl.constexpr          # number of elements per program
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load indices (int32), masked for bounds
    idxs = tl.load(inp_ptr + offsets, mask=mask, other=0)

    # For each expert id, compute the count in this block and atomically add to counts[expert_id]
    for j in range(NUM_EXPERTS):
        match = idxs == j            # boolean vector
        # Convert boolean to int32 (1 where True, 0 where False)
        match_i32 = match.to(tl.int32)
        # Sum across the block to get the count for expert j
        count_j = tl.sum(match_i32, axis=0)  # scalar int32
        # Atomically add this block's count to the global count for expert j
        tl.atomic_add(counts_ptr + j, count_j)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        """
        Triton-optimized version:
        - Flattens topk_idx, computes per-expert counts via a Triton kernel with block-wise reduction
          and a single atomic per bin per program.
        - Computes expert_offsets via torch.cumsum on the device (data-dependent part on GPU).
        - Keeps stable sort in PyTorch (data-independent on num_experts).
        """
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

        # Triton configuration: num_experts and block size
        NUM_EXPERTS = 256  # same as in the original code
        # Choose a reasonable block size; 1024 reduces kernel grid size while keeping good occupancy.
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        # Allocate counts on device (int32), initialized to zeros
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)

        # Run the histogram kernel
        _histogram_block_atomic_kernel[grid](
            flat,                          # input pointer
            N,                             # number of elements
            counts,                        # output counts
            NUM_EXPERTS=NUM_EXPERTS,      # constexpr for JIT specialization
            BLOCK=BLOCK,                   # constexpr for block size
            num_warps=4,                   # tuneable
        )

        # Compute expert offsets: inclusive prefix sum of counts, plus 0 at index 0
        # Keep everything on device; no host sync (no .item())
        expert_offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        running = 0  # int32 running sum on device? Triton loops don't fit here; use torch ops.
        # Use torch.cumsum on device to get inclusive prefix sums
        expert_offsets[1:] = counts.cumsum(dim=0)

        # Stable sort of flattened indices (keep in PyTorch, data-independent on num_experts)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
