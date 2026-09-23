import torch
import triton
import triton.language as tl


# Triton kernel: histogram of flattened indices (int32).
# For each element in flat, atomically increment counts[value].
# Assumes values are in [0, NUM_VALUES-1]. Here NUM_VALUES=256.
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    # For each lane i in this program, increment counts[vals[i]] if in bounds
    for i in range(BLOCK):
        v = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + v, 1)


# Triton kernel: compute inclusive prefix sum of counts -> prefix[v] = sum_{x<=v} counts[x]
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    # Inclusive scan: prefix[0] = counts[0]
    tl.store(prefix_ptr + 0, tl.load(counts_ptr + 0))
    # For i > 0: prefix[i] = prefix[i-1] + counts[i]
    for i in range(1, NUM_VALUES):
        prev = tl.load(prefix_ptr + (i - 1))
        cur = tl.load(counts_ptr + i)
        tl.store(prefix_ptr + i, prev + cur)


# Triton kernel: assemble expert_offsets from prefix:
# offsets[0] = 0; offsets[i+1] = prefix[i] for i in [0..NUM_VALUES-1]
@triton.jit
def assemble_offsets(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    tl.store(offsets_ptr + 0, 0)
    for i in range(NUM_VALUES):
        p = tl.load(prefix_ptr + i)
        tl.store(offsets_ptr + (i + 1), p)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on device and int32
        device = topk_idx.device
        # Flatten to 1D (view is a no-op if already contiguous)
        flat = topk_idx.contiguous().view(-1)
        M = flat.numel()

        NUM_VALUES = 256  # matches original run() logic
        # Allocate counts and offsets
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)

        # 1) Triton histogram
        BLOCK = 1024
        grid_hist = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid_hist](flat, counts, M, NUM_VALUES, BLOCK)

        # 2) Triton inclusive prefix sum
        prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES)

        # 3) Assemble offsets
        assemble_offsets[(1,)](prefix, offsets, NUM_VALUES)

        # Return expert offsets (int32, length NUM_VALUES+1)
        return offsets


def run(*args):
    return ModelNew()(*args)
