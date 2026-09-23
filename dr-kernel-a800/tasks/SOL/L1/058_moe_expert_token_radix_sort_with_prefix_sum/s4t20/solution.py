import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of indices (int32)
    ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # other=0 ensures masked lanes are 0
    ids = ids.to(tl.int32)

    # Accumulate counts per bin (unrolled loop)
    # counts_block[i] = number of occurrences of expert id == i within this block
    counts_block = tl.zeros((NUM_EXPERTS,), dtype=tl.int32)
    for i in range(NUM_EXPERTS):
        # Build a vector of (ids == i) and reduce to a scalar sum
        eq = ids == i  # vector boolean
        eq_i32 = eq.to(tl.int32)  # 1 where eq, else 0
        # Sum over the vector with a reduction to scalar
        # We do a simple loop to sum 0/1 values into scalar
        partial = tl.zeros((), dtype=tl.int32)  # scalar accumulator
        for j in range(BLOCK):
            # index j is a scalar; eq_i32[j] is a scalar 0/1; accumulate
            partial += eq_i32[j]
        counts_block[i] = partial

    # Atomically add block counts to global counts
    for i in range(NUM_EXPERTS):
        tl.atomic_add(counts_ptr + i, counts_block[i])


@triton.jit
def prefix_inclusive_scan_kernel(input_ptr, output_ptr, prefix_ptr, N, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    # Each program handles a block of indices and computes its inclusive prefix sum
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of counts
    vals = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.int32)
    # Compute inclusive scan for this block and write to output
    running = tl.zeros((), dtype=tl.int32)
    # We'll accumulate directly into output_ptr offsets
    for i in range(BLOCK):
        v = vals[i]
        running += v
        tl.store(output_ptr + offsets[i], running, mask=mask[i])

    # Compute the total of this block's values to pass up via atomic add
    total = tl.zeros((), dtype=tl.int32)
    for i in range(BLOCK):
        total += vals[i]
    # Atomically add this block's total to the prefix (exclusive) for the next block
    # prefix[pid] = sum of this block
    tl.atomic_add(prefix_ptr + pid, total)


def _triton_histogram(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    N = flat.numel()
    device = flat.device
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)

    BLOCK = 2048  # tile size; tuneable. 2048 works well for these sizes.
    grid = (triton.cdiv(N, BLOCK),)
    histogram_kernel[grid](flat, counts, N, NUM_EXPERTS=num_experts, BLOCK=BLOCK, num_warps=8)
    return counts


def _triton_inclusive_prefix_sum(counts: torch.Tensor) -> torch.Tensor:
    num_experts = counts.numel()
    device = counts.device
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    prefix = torch.zeros(1, dtype=torch.int32, device=device)  # will hold running prefix across blocks

    BLOCK = 1024  # scan block size
    grid = (triton.cdiv(num_experts, BLOCK),)
    # We need to pass an extra buffer for prefix to accumulate block totals.
    # Launch with a grid; Triton kernel will atomically add its block total to prefix.
    prefix_inclusive_scan_kernel[grid](counts, offsets[1:], prefix, num_experts, NUM_EXPERTS=num_experts, BLOCK=BLOCK, num_warps=8)
    # The offsets[0] remains 0; we don't modify it in the kernel. Return offsets.
    return offsets


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; the module will launch Triton kernels in forward.

    def forward(self, topk_idx: torch.Tensor):
        # Ensure device and contiguity
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()

        # Triton histogram of expert IDs
        num_experts = 256
        counts = _triton_histogram(flat, num_experts)

        # Triton inclusive prefix sum to produce expert_offsets
        expert_offsets = _triton_inclusive_prefix_sum(counts)

        # Stable sort of flattened indices (data-independent, kept in PyTorch)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
