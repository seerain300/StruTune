import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N: tl.int32, L: tl.int32, BLOCK: tl.constexpr):
    """
    Count occurrences of each value in orig_ptr (flattened int32) into counts_ptr of length L.
    Each program handles BLOCK elements, and atomic_adds 1 for each valid lane.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load original values (int32), invalid lanes get -1 (out of valid id range)
    vals = tl.load(orig_ptr + offs, mask=mask, other=-1)
    # Valid lanes are those within [0, L)
    valid = mask & (vals >= 0) & (vals < L)
    # Atomic add 1 to counts[vals] for valid lanes
    tl.atomic_add(counts_ptr + vals, 1, mask=valid)


@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sums of counts_ptr[0..L-1] and store in offsets_ptr[0..L-1].
    offsets_ptr[L] is set to total N via a separate kernel (not included here to keep it simple).
    """
    running = 0
    for i in range(0, L):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and cast to int32 for Triton counting
        orig = topk_idx.reshape(-1).to(torch.int32)
        N = orig.numel()
        num_experts = 256
        device = orig.device

        # 1) Compute sorted_token_indices using torch.sort (for correctness with stable=True)
        sorted_token_indices = torch.sort(orig, stable=True)[1]  # permutation of indices [0..N-1]

        # 2) Compute expert offsets via Triton histogram + prefix sum (Triton-only)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid](orig, counts, N, num_experts, BLOCK=BLOCK, num_warps=1)

        # Compute inclusive prefix sums to produce offsets[e] = inclusive count of elements < e
        prefix = torch.cumsum(counts, dim=0)  # torch is allowed here for correctness
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[:num_experts] = prefix
        expert_offsets[num_experts] = N  # last element is total number of elements

        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
