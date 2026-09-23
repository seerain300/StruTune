import torch
import triton
import triton.language as tl


# Kernel 1: Histogram of flattened indices (int32) into counts[value].
# Assumes indices are in [0, NUM_VALUES-1]. We use NUM_VALUES=256 to match original.
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M: tl.constexpr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        v = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + v, 1)


# Kernel 2: Inclusive prefix sum of counts -> prefix[v] = sum_{x<=v} counts[x].
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    # Initialize prefix[0] = 0
    tl.atomic_add(prefix_ptr + 0, 0)
    i = 0
    while i < NUM_VALUES:
        count_i = tl.load(counts_ptr + i)
        # prefix[i+1] = prefix[i] + count_i
        tl.atomic_add(prefix_ptr + (i + 1), tl.load(prefix_ptr + i))
        tl.atomic_add(prefix_ptr + (i + 1), count_i)
        i += 1


# Kernel 3: Assemble expert_offsets from prefix:
# offsets[0] = 0; offsets[i+1] = prefix[i] for i in [0..NUM_VALUES-1].
@triton.jit
def assemble_offsets(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    tl.store(offsets_ptr + 0, 0)
    i = 0
    while i < NUM_VALUES:
        p = tl.load(prefix_ptr + i)
        tl.store(offsets_ptr + (i + 1), p)
        i += 1


# Kernel 4: Stable counting sort by expert index. Produces sorted_token_indices permutation.
# Grid: (NUM_VALUES,) — one program per expert value e in [0..NUM_VALUES-1].
@triton.jit
def sort_stable_by_expert_kernel(flat_ptr, sorted_ptr, M: tl.constexpr, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0)  # expert value
    # Phase A: count less and equals
    less = tl.zeros((), dtype=tl.int32)
    equals = tl.zeros((), dtype=tl.int32)
    j = 0
    while j < M:
        v = tl.load(flat_ptr + j)
        if v < e:
            less += 1
        elif v == e:
            equals += 1
        j += 1

    # Phase B: compute stable insertion positions for each occurrence of e.
    # For each j, position = less + (number of e that appear before j in original order).
    start = less  # base insertion index for this value
    processed = tl.zeros((), dtype=tl.int32)  # how many equal elements have been written so far
    j = 0
    while j < M:
        v = tl.load(flat_ptr + j)
        if v == e:
            # Stable tie-break: earlier j comes first
            pos = start + (equals - processed - 1)
            # Write index j at position pos
            tl.store(sorted_ptr + pos, j)
            processed += 1
        j += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Match original code: use 256 possible values
        self.NUM_VALUES = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure Triton execution on CUDA
        if topk_idx.device.type != "cuda":
            topk_idx = topk_idx.to("cuda")

        # Flatten indices to 1D
        flat = topk_idx.reshape(-1)
        M = flat.numel()

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=topk_idx.device)
        counts = torch.zeros(self.NUM_VALUES, dtype=torch.int32, device=topk_idx.device)
        prefix = torch.empty(self.NUM_VALUES, dtype=torch.int32, device=topk_idx.device)
        expert_offsets = torch.empty(self.NUM_VALUES + 1, dtype=torch.int32, device=topk_idx.device)

        # Launch histogram kernel
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(M, BLOCK_HIST),)
        histogram_kernel[grid_hist](flat, counts, M, self.NUM_VALUES, BLOCK_HIST)

        # Launch prefix sum kernel
        prefix_sum_kernel[(1,)](counts, prefix, self.NUM_VALUES)

        # Launch assemble offsets kernel
        assemble_offsets[(1,)](prefix, expert_offsets, self.NUM_VALUES)

        # Launch stable sort-by-expert kernel: one program per expert value
        grid_sort = (self.NUM_VALUES,)
        sort_stable_by_expert_kernel[grid_sort](flat, sorted_token_indices, M, self.NUM_VALUES, BLOCK_HIST)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
