import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M: tl.int32, N: tl.int32, BLOCK: tl.constexpr):
    # Each program handles one value v in [0..N-1]
    v = tl.program_id(0)
    # Initialize count for this value
    count = tl.zeros((), dtype=tl.int32)
    # Scan over original vector in tiles of BLOCK
    for i in range(0, 1024):  # MAX_M tile; i < M ensures valid range
        idx = i
        valid = idx < M
        val = tl.load(original_ptr + idx, mask=valid, other=0)
        eq = val == v
        # atomic add to global counts[v]
        tl.atomic_add(counts_ptr + v, eq.to(tl.int32))


@triton.jit
def cumsum_kernel(counts_ptr, prefix_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Compute inclusive prefix sum of counts[0..N-1] and write to prefix_ptr[0..N-1]
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, N):
        acc += tl.load(counts_ptr + i)
        tl.store(prefix_ptr + i, acc)


@triton.jit
def assemble_offsets_kernel(prefix_ptr, offsets_ptr, N: tl.int32):
    # offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    # offsets[i+1] = prefix[i] for i in [0..N-1]
    # Note: this kernel assumes N is passed and valid
    for i in range(0, N):
        tl.store(offsets_ptr + (i + 1), tl.load(prefix_ptr + i))


@triton.jit
def stable_permutation_kernel(original_ptr, sorted_ptr, prefix_ptr, M: tl.int32, N: tl.int32, BLOCK: tl.constexpr):
    # Stable permutation: place each value v, then place indices i where original[i] == v
    # in order of original positions, using prefix[v-1] for initial offset and counting equal-before.
    for v in range(N):
        number_of_less = 0
        if v > 0:
            number_of_less = tl.load(prefix_ptr + (v - 1))
        # For each index i, if original[i] == v, find number_of_equal_before_i by scanning previous indices
        for i in range(0, 1024):  # tile of indices
            idx = i
            valid_i = idx < M
            original_i = tl.load(original_ptr + idx, mask=valid_i, other=0)
            eq = valid_i & (original_i == v)
            neqb = 0
            # Count equal before idx by scanning previous indices
            for j in range(0, 1024):
                original_j = tl.load(original_ptr + j)
                # j < idx is equivalent to j < i when i is a loop index
                if (j < idx) and (original_j == v):
                    neqb += 1
            # Store pos for valid eq
            pos = number_of_less + neqb
            tl.store(sorted_ptr + idx, pos, mask=valid_i & eq)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is CUDA int32
        assert topk_idx.is_cuda, "Input must be on CUDA device."
        # Flatten and ensure contiguous int32
        original = topk_idx.contiguous().view(-1)
        # The original code uses int32; ensure dtype
        if original.dtype != torch.int32:
            original = original.to(torch.int32)
        M = original.numel()
        device = original.device

        # Number of experts per token is passed in axes; in the given workloads it is 256
        num_experts_per_tok = 256
        N = num_experts_per_tok

        # 1) Histogram of values in [0..N-1]
        counts = torch.zeros(N, dtype=torch.int32, device=device)
        # Launch histogram kernel: grid size = N (one program per value)
        grid_histogram = (N,)
        histogram_kernel[grid_histogram](original, counts, M, N, BLOCK=1024)

        # 2) Inclusive prefix sum of counts
        prefix = torch.empty(N, dtype=torch.int32, device=device)
        grid_cumsum = (N,)
        cumsum_kernel[grid_cumsum](counts, prefix, N, BLOCK=1024)

        # 3) Assemble expert offsets
        offsets = torch.empty(N + 1, dtype=torch.int32, device=device)
        # offsets[0] = 0; offsets[i+1] = prefix[i]
        assemble_offsets_kernel[(1,)](prefix, offsets, N)  # single program writes

        # 4) Stable permutation: sorted_token_indices
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        grid_perm = (1,)
        stable_permutation_kernel[grid_perm](original, sorted_token_indices, prefix, M, N, BLOCK=1024)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
