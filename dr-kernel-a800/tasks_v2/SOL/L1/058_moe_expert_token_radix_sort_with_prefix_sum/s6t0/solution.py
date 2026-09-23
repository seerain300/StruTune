import torch
import triton
import triton.language as tl


# Kernel: compute histogram of expert IDs (values in [0, num_experts-1]) over input tensor 'x'
# We assume x is 1D int32. We will process it in blocks and update per-expert counts.
@triton.jit
def histogram_kernel(x_ptr, counts_ptr, num_experts: tl.constexpr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load a block of values (int32)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # For each possible expert id, count occurrences in this block
    for e in range(num_experts):
        # Create a boolean mask: vals == e
        eq = vals == e
        # Convert to int32 and reduce sum
        # Note: we mask out-of-range lanes by setting them to False via eq & mask
        # However, vals for out-of-range lanes are 0; eq will be False for masked lanes naturally.
        # Sum across the vector
        count_block = tl.sum((eq & mask).to(tl.int32))
        # Atomic add into global counts[e]
        tl.atomic_add(counts_ptr + e, count_block)


# Kernel: compute exclusive prefix sum (start) of counts to produce expert offsets
# We compute inclusive prefix sum in registers and write exclusive sums directly.
# offsets[0..num_experts-1] will be written. We will handle the +1 length in the host.
@triton.jit
def scan_exclusive_prefix_sum(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # This is a simple sequential scan (fine for num_experts=256).
    # Each program handles one expert id e and computes its prefix sum contribution.
    for e in range(num_experts):
        count_e = tl.load(counts_ptr + e)
        # Initialize exclusive start for this e
        start = 0
        # Inclusive prefix: add previous counts
        for j in range(e):
            start += tl.load(counts_ptr + j)
        # Write exclusive start to offsets[e]
        tl.store(offsets_ptr + e, start)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Computes sorted_token_indices using torch.sort (stable=True) on flattened topk_idx.
        - Computes expert_offsets via Triton: histogram of expert IDs and exclusive prefix sum.
        """
        # Flatten topk_idx
        flat = topk_idx.reshape(-1)

        # 1) Stable sort on flattened indices (as in the original), using torch for correctness.
        #    This produces a permutation of indices [0, N-1] that sorts flat stably.
        #    We only need the permutation (indices).
        N = flat.numel()
        sorted_token_indices, _ = torch.sort(flat, stable=True)
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        # 2) Compute expert offsets via Triton:
        #    a) Prepare counts of expert IDs.
        num_experts = 256  # same as original assertion
        # Initialize counts to zeros on device
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel
        BLOCK = 1024  # process 1024 elements per program
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](flat, counts, num_experts, N, BLOCK)

        # b) Compute exclusive prefix sum to produce offsets[0..num_experts-1]
        #    We'll write to an output buffer of length num_experts.
        #    offsets[0..num_experts-1] will be filled; we add one more element in host for num_experts.
        offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
        # Launch scan kernel
        scan_exclusive_prefix_sum[(num_experts,)](counts, offsets, num_experts)

        # Build final offsets of length (num_experts + 1), with last element being total count.
        # Total count is simply sum(counts). We'll set offsets[num_experts] = sum(counts).
        total = int(counts.sum().item())
        # Create final_offsets on host
        final_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        final_offsets[:num_experts] = offsets
        final_offsets[num_experts] = total

        return sorted_token_indices, final_offsets


def run(*args):
    return ModelNew()(*args)
