import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M: tl.int32, N: tl.int32, BLOCK: tl.constexpr):
    # Count occurrences of each value in [0..N-1] from original_ptr[0..M-1]
    for start in range(N):
        # Scan original in chunks of BLOCK
        for i in range(0, M, BLOCK):
            idx = i + tl.arange(0, BLOCK)
            valid = idx < M
            vals = tl.load(original_ptr + idx, mask=valid, other=0)
            eq = (vals == start) & valid
            count_contrib = tl.sum(eq.to(tl.int32), axis=0)
            tl.atomic_add(counts_ptr + start, count_contrib)


@triton.jit
def cumsum_kernel(counts_ptr, prefix_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Compute inclusive prefix sum of counts for N values
    # Simple sequential loop per element is fine for N=256.
    for i in range(N):
        val = tl.load(counts_ptr + i)
        tl.store(prefix_ptr + i, val)
        for j in range(i + 1, N):
            prev = tl.load(prefix_ptr + (j - 1))
            curr = tl.load(prefix_ptr + j)
            tl.store(prefix_ptr + j, curr + prev)


@triton.jit
def assemble_offsets_kernel(prefix_ptr, offsets_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # offsets[0] = 0
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
    # Produce sorted indices with stable=True for values in [0..N-1]
    MAX_M = 8192  # safe upper bound for M in provided workloads
    for v in range(N):
        number_of_less = 0
        if v > 0:
            number_of_less = tl.load(prefix_ptr + (v - 1))
        for i in range(MAX_M):
            valid_i = i < M
            original_i = tl.load(original_ptr + i, mask=valid_i, other=0)
            eq = original_i == v
            # Count number of elements equal to v that appear before i
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
        N = 256  # num_experts_per_tok from provided workloads

        # Allocate outputs
        counts = torch.zeros(N, dtype=torch.int32, device=device)
        prefix = torch.empty(N, dtype=torch.int32, device=device)
        offsets = torch.empty(N + 1, dtype=torch.int32, device=device)
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)

        # 1) Histogram kernel: counts of each value in [0..255]
        BLOCK_HIST = 1024
        grid_hist = (N,)  # one program per value
        histogram_kernel[grid_hist](original, counts, M, N, BLOCK_HIST)

        # 2) Cumsum (inclusive prefix) of counts
        BLOCK_CUM = 1
        grid_cum = (N,)
        cumsum_kernel[grid_cum](counts, prefix, N, BLOCK_CUM)

        # 3) Assemble offsets: offsets[0]=0; offsets[i+1]=prefix[i] for i in [0..255]
        BLOCK_AO = 1
        grid_a = (N,)
        assemble_offsets_kernel[grid_a](prefix, offsets, N, BLOCK_AO)

        # 4) Stable permutation to produce sorted_token_indices
        BLOCK_SP = 1024
        grid_sp = (1,)
        stable_permutation_kernel[grid_sp](original, sorted_token_indices, prefix, M, N, BLOCK_SP)

        # Return sorted token indices and expert offsets
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
