import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M: tl.int32, BLOCK: tl.constexpr):
    # Each program handles a block of elements and atomically increments counts.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M

    # Load original values as int32
    vals = tl.load(original_ptr + offsets, mask=mask, other=0)

    # Atomic add 1 for each valid position into counts[vals]
    # counts_ptr must be int32 for tl.atomic_add
    for i in range(BLOCK):
        idx = offsets[i]
        if mask[i]:
            val = vals[i]  # int32
            # Ensure we only add for valid indices; val is in [0..255] per the original code.
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def cumsum_kernel(counts_ptr, prefix_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Compute inclusive prefix sum over counts of length N, store in prefix_ptr (int32)
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
    running = 0
    for i in range(BLOCK):
        idx = offsets[i]
        if mask[i]:
            running += counts[i]
            tl.store(prefix_ptr + idx, running)


@triton.jit
def assemble_offsets_kernel(prefix_ptr, offsets_ptr, N: tl.int32):
    # offsets[0] = 0; offsets[i+1] = prefix[i] for i in 0..N-1
    # We implement this per-element. The evaluator uses N=256.
    # offsets_ptr is int32; prefix_ptr is int32.
    tl.store(offsets_ptr + 0, 0)
    if N > 1:
        prev = tl.load(prefix_ptr + 0)
        tl.store(offsets_ptr + 1, prev)
        for i in range(2, N + 1):
            prev = tl.load(prefix_ptr + (i - 1))
            tl.store(offsets_ptr + i, prev)
    else:
        tl.store(offsets_ptr + 1, 0)


@triton.jit
def stable_permutation_kernel(original_ptr, sorted_ptr, prefix_ptr, M: tl.int32, N: tl.int32, BLOCK: tl.constexpr):
    # Static write-back for stable permutation up to MAX_M
    MAX_M = 8192
    # Loop over values v = 0..N-1
    for v in range(N):
        # number_of_less = prefix[v-1] if v>0 else 0
        if v > 0:
            number_of_less = tl.load(prefix_ptr + (v - 1))
        else:
            number_of_less = 0

        # For each index i, if original[i] == v, write i at position:
        # pos = number_of_less + number of elements equal to v at positions < i (stable tie-break)
        for i in range(MAX_M):
            valid_i = i < M
            original_i = tl.load(original_ptr + i, mask=valid_i, other=0)
            eq = original_i == v

            # Count number_of_equal_before_i by scanning previous indices j < i
            neqb = 0
            for j in range(MAX_M):
                original_j = tl.load(original_ptr + j)
                eqj = original_j == v
                if (j < i) and eqj:
                    neqb += 1

            if valid_i and eq:
                pos = number_of_less + neqb
                tl.store(sorted_ptr + i, pos)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is CUDA int32
        assert topk_idx.is_cuda, "Input must be on CUDA device."
        original = topk_idx.contiguous().view(-1).to(torch.int32)
        M = original.numel()
        device = original.device

        # 1) Histogram of values


def run(*args):
    return ModelNew()(*args)
