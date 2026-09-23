import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(inp_ptr, N, out_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute histogram of 1D int32/int64 indices in inp_ptr[0:N] into out_ptr[0:num_experts].
    out_ptr[i] = number of occurrences of index i.
    Uses atomic_add to handle multiple elements mapping to the same bin in parallel.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load indices; cast to int64 to be safe for address arithmetic
    idx64 = tl.load(inp_ptr + offsets, mask=mask, other=0).to(tl.int64)

    # For masked lanes, idx is 0 so they don't contribute
    for j in range(BLOCK):
        i = offsets[j]
        valid = i < N
        # If valid, atomically add 1 to out_ptr[idx]
        idx = idx64[j]
        tl.atomic_add(out_ptr + idx, 1, mask=valid)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Triton-optimized version:
        - Flattens topk_idx, computes histogram in Triton, then creates expert_offsets via torch.cumsum.
        - Keeps the stable sort in PyTorch as in the original code for simplicity and correctness.
        """
        # The original signature: run(topk_idx: torch.Tensor) with no other args.
        # Here we assume the only argument is the tensor of expert indices.
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")

        topk_idx = args[0]
        if not topk_idx.is_cuda:
            raise RuntimeError("topk_idx must be on CUDA device to use Triton.")

        # Ensure it's contiguous and flatten
        topk_idx = topk_idx.contiguous()
        flat = topk_idx.reshape(-1)

        # Stable sort of tokens by expert id (as in the original)
        # Using PyTorch for simplicity; keeping the same behavior as the original.
        _, sorted_token_indices = flat.sort(stable=True)

        # Triton histogram for counts per expert
        num_experts = 256  # match the original code assumption
        N = flat.numel()
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch Triton kernel
        # Choose a block size; 1024 is a good default for small to medium N.
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_kernel[grid](
            flat,  # inp_ptr
            N,     # N (runtime)
            counts,  # out_ptr
            num_experts=num_experts,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Compute expert offsets via cumsum (inclusive): prefix[i] = sum_{j=0..i} counts[j]
        # +1 for the final sentinel as in the original code
        # Note: counts may contain zeros if some experts have no tokens assigned.
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        # Inclusive prefix sum
        running = 0
        for i in range(num_experts):
            running += counts[i]
            expert_offsets[i + 1] = running

        # Return same outputs as original: (sorted_token_indices, expert_offsets)
        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
