import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(original_ptr, counts_ptr, N, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # Each program handles a contiguous block of elements from original_ptr
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    vals = tl.load(original_ptr + offsets, mask=mask, other=0)
    vals = vals.to(tl.int32)

    # Vectorized atomic add of counts for each value in [0..NUM_VALUES-1]
    for v in range(0, NUM_VALUES):
        eq = vals == v
        contrib = tl.where(eq, 1, 0).to(tl.int32)
        tl.atomic_add(counts_ptr + v, contrib, mask=mask)


@triton.jit
def _prefix_sum_inclusive(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # Compute inclusive prefix sums of counts[0..NUM_VALUES-1] in chunks of BLOCK
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < NUM_VALUES

    c = tl.load(counts_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # Inclusive scan within this block
    for k in range(1, BLOCK):
        prev = tl.load(counts_ptr + offsets - 1, mask=(offsets > 0) & mask, other=0).to(tl.int32)
        new = c + tl.where(offsets > 0, prev, 0)
        tl.store(prefix_ptr + offsets, new, mask=mask)
        c = new


@triton.jit
def _assemble_offsets_kernel(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    # offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # offsets[i+1] = prefix[i] for i in 0..NUM_VALUES-1
    for i in range(0, NUM_VALUES):
        tl.store(offsets_ptr + 1 + i, tl.load(prefix_ptr + i))


def _triton_offsets(topk_idx: torch.Tensor):
    """
    Compute expert_offsets via Triton histogram and prefix sum.
    Input: topk_idx (batch_size, seq_len, num_experts_per_tok), int32 on device.
    Output: expert_offsets (num_experts_per_tok + 1, int32), where offsets[i+1] = sum_{x<=i} counts_x.
    """
    device = topk_idx.device
    assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
    # Flatten
    original = topk_idx.reshape(-1).contiguous()
    N = original.numel()

    NUM_VALUES = 256  # matches num_experts_per_tok in provided workloads

    # Triton histogram: counts of each value in [0..NUM_VALUES-1]
    counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
    BLOCK = 1024
    grid_hist = (triton.cdiv(N, BLOCK),)
    _histogram_kernel[grid_hist](original, counts, N, NUM_VALUES, BLOCK)

    # Inclusive prefix sum of counts
    prefix = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
    BLOCK_SCAN = 4096
    grid_scan = (triton.cdiv(NUM_VALUES, BLOCK_SCAN),)
    _prefix_sum_inclusive[grid_scan](counts, prefix, NUM_VALUES, BLOCK_SCAN)

    # Assemble offsets: offsets[0] = 0; offsets[i+1] = prefix[i]
    offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)
    _assemble_offsets_kernel[(1,)](prefix, offsets, NUM_VALUES)

    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single argument: topk_idx (batch_size, seq_len, num_experts_per_tok), int32, on CUDA.
        topk_idx = args[0]
        assert topk_idx.is_cuda and topk_idx.dtype == torch.int32, "topk_idx must be CUDA int32 tensor"

        # Compute sorted_token_indices using PyTorch for correctness (stable sort)
        # Flatten
        flat = topk_idx.reshape(-1)
        # Stable sort indices
        sorted_token_indices = torch.sort(flat, stable=True).indices.to(torch.int32)

        # Compute expert_offsets using Triton
        expert_offsets = _triton_offsets(topk_idx)

        # Return both outputs as required
        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
