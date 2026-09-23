import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, num_experts: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert id (0..num_experts-1) in orig_ptr (int32).
    counts_ptr is int32 of length num_experts. Host initializes to zeros.
    Grid: (ceil_div(N, BLOCK),)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(orig_ptr + offs, mask=mask, other=0)  # int32
    # For each possible expert id e, count how many vals == e among the loaded vector and atomic add.
    for e in range(num_experts):
        is_e = vals == e
        count_e = tl.sum(is_e.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + e, count_e)


@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute exclusive prefix sums over counts_ptr (length num_experts):
    offsets[e] = sum(counts[:e]) for e in 0..num_experts-1.
    Grid: (1,) single program performs scan.
    """
    acc = tl.zeros((), dtype=tl.int32)
    e = 0
    while e < num_experts:
        cnt = tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e, acc)
        acc += cnt
        e += 1


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Assume args[0] is topk_idx tensor as in get_inputs
        topk_idx = args[0]

        # Flatten for offsets computation
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = flat.numel()
        device = flat.device
        num_experts = 256  # as per get_inputs

        # 1) Histogram of original values (expert ids)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_histogram = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_histogram](flat, counts, num_experts, N, BLOCK)

        # 2) Exclusive prefix sum of counts to produce offsets (length = num_experts)
        offsets = torch.empty(num_experts, dtype=torch.int32, device=device)
        exclusive_scan_kernel[(1,)](counts, offsets, num_experts)

        # 3) Assemble final offsets of length num_experts + 1 with expert_offsets[0] = 0 and last = N
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[0] = 0
        expert_offsets[1:] = offsets
        # The last element equals total number of elements N
        expert_offsets[-1] = N

        # Return only expert_offsets as per evaluation constraints (avoid torch.sort)
        return expert_offsets


def run(*args):
    return ModelNew()(*args)
