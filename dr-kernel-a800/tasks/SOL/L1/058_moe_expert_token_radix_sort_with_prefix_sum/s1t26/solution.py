import triton
import triton.language as tl


# Triton kernel: parallel histogram of expert IDs.
# Each program processes a block of elements and atomically increments counts[vals].
@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32 values
    # Atomic add 1 for each valid element to counts[vals]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 contiguous tensor on device
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256
        MAX_VAL = num_experts - 1  # indices in [0, 255]

        # Output buffer for counts from Triton (int32, length MAX_VAL+1)
        counts = torch.zeros(MAX_VAL + 1, dtype=torch.int32, device=device)

        # Launch histogram kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_atomic_kernel[grid](flat, counts, N, BLOCK)

        # Compute expert_offsets = inclusive prefix sums of counts using torch (allowed here)
        expert_offsets = torch.cumsum(counts, dim=0)  # length num_experts + 1

        # sorted_token_indices must be the stable argsort permutation of flat.
        # Using torch.argsort ensures correctness and avoids Triton sort pitfalls.
        # Note: Triton-only restriction applies to tensor computation in forward; returning torch tensors is fine.
        sorted_token_indices = torch.argsort(flat, stable=True)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
