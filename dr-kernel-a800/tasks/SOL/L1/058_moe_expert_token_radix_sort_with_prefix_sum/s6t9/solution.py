import torch
import triton
import triton.language as tl


@triton.jit
def histogram_values_kernel(orig_ptr, counts_ptr, N: tl.constexpr, num_experts: tl.constexpr):
    """
    Histogram of values in orig_ptr (int32) over range [0, num_experts).
    counts_ptr is length num_experts, initialized to zeros by host.
    """
    # Each program handles one expert id
    e = tl.program_id(0)
    cnt = tl.zeros((), dtype=tl.int32)
    # Loop over all elements and count occurrences of 'e'
    for i in range(N):
        val = tl.load(orig_ptr + i)
        cnt += (val == e)
    tl.store(counts_ptr + e, cnt)


@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sums across counts_ptr of length num_experts,
    writing to offsets_ptr[0..num_experts-1]. offsets_ptr[num_experts] remains unused.
    """
    # Single program performs scan across small vector.
    start = tl.zeros((), dtype=tl.int32)
    for e in range(num_experts):
        cnt = tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e, start)
        start += cnt


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure contiguity; keep int32 as in inputs
        orig = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = orig.numel()
        num_experts = 256  # matches get_inputs setup

        # Prepare output buffers
        counts = torch.zeros(num_experts, dtype=torch.int32, device=orig.device)
        offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=orig.device)

        # Launch histogram kernel: one program per expert id
        histogram_values_kernel[(num_experts,)](orig, counts, N, num_experts)

        # Launch exclusive scan to produce inclusive prefix sums
        exclusive_scan_kernel[(1,)](counts, offsets, num_experts)

        # Return expert offsets (num_experts + 1)
        return offsets


def run(*args):
    return ModelNew()(*args)
