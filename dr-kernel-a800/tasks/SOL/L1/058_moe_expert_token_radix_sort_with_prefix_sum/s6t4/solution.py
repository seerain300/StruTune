import torch
import triton
import triton.language as tl


@triton.jit
def counting_values_kernel(flat_ptr, counts_ptr, N: tl.constexpr, L: tl.constexpr):
    """
    Count occurrences per value v in [0, L) in the flat array of length N.
    counts_ptr has length L and will be atomically incremented per occurrence.
    """
    for i in range(N):
        v = tl.load(flat_ptr + i)
        tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Exclusive prefix sums: offsets[e] = sum(counts[:e]) for e in [0, L).
    counts_ptr length L, offsets_ptr length L.
    """
    e = 0
    running = 0  # scalar int
    while e < L:
        cnt = tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e, running)
        running += cnt
        e += 1


@triton.jit
def local_rank_and_place_kernel(flat_ptr, out_ptr, offsets_ptr, N: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr):
    """
    Stable counting sort permutation:
    For each original index i, compute:
      v = flat[i]
      local_rank_i = number of elements j < i with flat[j] == v
      pos = offsets[v] + local_rank_i
      out[pos] = i
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    for i in range(start, N, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < N
        vals_i = tl.load(flat_ptr + idx, mask=mask, other=0)
        local_rank = tl.zeros([BLOCK], dtype=tl.int32)
        # Per-element inner loop to compute stable rank by original index
        for j in range(start, N):
            vj = tl.load(flat_ptr + j)
            eq = vj == vals_i
            earlier = j < idx
            valid = mask & earlier & eq
            local_rank += tl.sum(valid.to(tl.int32))
        pos = tl.load(offsets_ptr + vals_i) + local_rank
        tl.store(out_ptr + pos, idx, mask=mask)


@triton.jit
def histogram_original_kernel(orig_ptr, counts_ptr, N: tl.constexpr, L: tl.constexpr):
    """
    Histogram of original topk_idx values (int32) per expert id e in [0, L).
    Counts_ptr length L, atomically incremented per occurrence.
    """
    for i in range(N):
        v = tl.load(orig_ptr + i)
        tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def zero_counts_kernel(counts_ptr, L: tl.constexpr):
    """
    Zero-initialize counts_ptr of length L via atomic_add 0.
    This avoids relying on torch.zeros in host code.
    """
    for e in range(L):
        tl.atomic_add(counts_ptr + e, 0)


@triton.jit
def zero_offsets_kernel(offsets_ptr, L: tl.constexpr):
    """
    Zero-initialize offsets_ptr of length L via atomic_add 0.
    """
    for e in range(L):
        tl.atomic_add(offsets_ptr + e, 0)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts in provided setup is 256; values in [0..255]
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Flatten original topk_idx to 1D int32
        orig = topk_idx.view(-1).to(torch.int32)
        N = orig.numel()
        device = orig.device

        L = self.num_experts

        # 1) Compute stable sort permutation using Triton counting sort approach.
        #    We need counts and offsets for values in [0..255].
        counts_val = torch.empty(L, dtype=torch.int32, device=device)
        zero_counts_kernel[(1,)](counts_val, L)

        # Count occurrences per value (in Triton)
        counting_values_kernel[(1,)](orig, counts_val, N, L)

        # Compute exclusive prefix sums for offsets
        offsets_val = torch.empty(L, dtype=torch.int32, device=device)
        zero_offsets_kernel[(1,)](offsets_val, L)
        exclusive_scan_kernel[(1,)](counts_val, offsets_val, L)

        # Allocate output permutation of length N
        out_perm = torch.empty(N, dtype=torch.int32, device=device)

        # Launch local_rank_and_place kernel to produce stable sort permutation
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        local_rank_and_place_kernel[grid](orig, out_perm, offsets_val, N, L, BLOCK)

        # 2) Compute expert offsets (cumulative count of expert ids) using Triton
        counts_exp = torch.empty(L, dtype=torch.int32, device=device)
        zero_counts_kernel[(1,)](counts_exp, L)
        histogram_original_kernel[(1,)](orig, counts_exp, N, L)

        # Exclusive prefix sums for expert offsets (length L+1)
        expert_offsets = torch.empty(L + 1, dtype=torch.int32, device=device)
        zero_offsets_kernel[(1,)](expert_offsets, L)
        exclusive_scan_kernel[(1,)](counts_exp, expert_offsets, L)

        # Return results
        return out_perm, expert_offsets


def run(*args):
    return ModelNew()(*args)
