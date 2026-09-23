import torch
import triton
import triton.language as tl


# Triton histogram kernel: counts[v] = number of times v appears in orig (int32)
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    # For each value v in [0..L-1], count occurrences among vals.
    for v in range(L):
        eq = vals == v
        per_lane = tl.where(eq, 1, 0)  # int32 vector
        incr = tl.sum(per_lane, axis=0)  # scalar count for this program's lanes
        tl.atomic_add(counts_ptr + v, incr)


# Triton exclusive prefix-sum kernel to compute bases for counting sort
# bases_out[i] = sum_{w < i} counts[w], for i in [0..L-1]
@triton.jit
def exclusive_scan_bases_kernel(counts_ptr, bases_out_ptr, L: tl.constexpr):
    running = 0
    for i in range(L):
        bases_out_ptr[i] = running
        running += counts_ptr[i]


# Triton counting-sort permutation kernel for distinct values in [0..L-1]
# It writes sorted_token_indices_out[base + lane] = flat_index for each element,
# where base = bases_out[vals[lane]]. Since values are distinct, no overlap occurs.
@triton.jit
def counting_sort_perm_kernel(orig_ptr, bases_out_ptr, sorted_token_indices_out_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    idx = offsets  # original flat indices [0..N-1]
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)  # expert ids in [0..L-1]
    # Determine base position for each value v: base = bases_out[v]
    for v in range(L):
        base = bases_out_ptr[v]
        eq = vals == v
        # Write idx for all lanes with value v at positions base + lane
        tl.store(sorted_token_indices_out_ptr + base + offsets, idx, mask=mask & eq)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch, seq_len, num_experts_per_tok), int32 on device
        assert topk_idx.is_cuda and topk_idx.dtype == torch.int32
        orig = topk_idx.contiguous().view(-1)  # 1D flat int32
        N = orig.numel()

        # Constants
        L = 256  # num_experts as per original code

        # Output buffers
        counts = torch.zeros(L, dtype=torch.int32, device=orig.device)
        bases_out = torch.empty(L, dtype=torch.int32, device=orig.device)
        sorted_token_indices_out = torch.empty(N, dtype=torch.int32, device=orig.device)

        # Launch histogram kernel
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_kernel[grid_hist](orig, counts, N, L, BLOCK_HIST)

        # Compute bases (exclusive prefix sum) via Triton
        exclusive_scan_bases_kernel[(1,)](counts, bases_out, L)

        # Launch counting sort permutation kernel
        BLOCK_SORT = 1024
        grid_sort = (triton.cdiv(N, BLOCK_SORT),)
        counting_sort_perm_kernel[grid_sort](orig, bases_out, sorted_token_indices_out, N, L, BLOCK_SORT)

        # expert_offsets: bases_out concatenated with total N at the end
        offsets = torch.empty(L + 1, dtype=torch.int32, device=orig.device)
        offsets[:-1] = bases_out
        offsets[-1] = N

        return sorted_token_indices_out, offsets


def run(*args):
    return ModelNew()(*args)
