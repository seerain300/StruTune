import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, num_experts: tl.constexpr, N, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert id (0..num_experts-1) in flat_ptr (int32).
    counts_ptr is int32 of length num_experts, zero-initialized by host.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    for e in range(num_experts):
        is_e = vals == e
        count_e = tl.sum(is_e.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + e, count_e)


@triton.jit
def scan_exclusive_values_kernel(counts_ptr, offsets_ptr, N: tl.constexpr):
    """
    Compute exclusive prefix sum over counts_ptr (length N) and write to offsets_ptr (length N).
    offset[v] = sum(counts[:v]) for v in 0..N-1.
    """
    start = tl.zeros((), dtype=tl.int32)
    v = 0
    while v < N:
        cnt = tl.load(counts_ptr + v)
        tl.store(offsets_ptr + v, start)
        start += cnt
        v += 1


@triton.jit
def count_values_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Count occurrences of each value (0..N-1) in flat_ptr (int32).
    counts_ptr is int32 of length N, zero-initialized by host.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    for v in range(N):
        is_v = vals == v
        count_v = tl.sum(is_v.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + v, count_v)


@triton.jit
def scan_exclusive_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Exclusive prefix sum over counts_ptr (length num_experts) and write to offsets_ptr (length num_experts).
    offset[e] = sum(counts[:e]) for e in 0..num_experts-1.
    """
    start = tl.zeros((), dtype=tl.int32)
    e = 0
    while e < num_experts:
        cnt = tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e, start)
        start += cnt
        e += 1


@triton.jit
def rank_atomic_kernel(flat_ptr, rank_ptr, N: tl.constexpr):
    """
    Compute stable local rank for each value v: rank[v] = number of elements with value v and index < current index.
    We iterate over indices in increasing order; for each i, read value v=flat[i] and atomically increment rank[v].
    rank_ptr is int32 of length N, zero-initialized by host.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # iterate per lane, atomically add 1 to rank[flat[offs]]
    for i in range(N):
        # compute masked load: only valid lanes participate; scalar loop over N is fine.
        val_i = tl.load(flat_ptr + i, mask=True, other=0)
        # increment rank at position i for value val_i
        tl.atomic_add(rank_ptr + val_i, 1)


@triton.jit
def placement_kernel(flat_ptr, out_ptr, rank_ptr, offsets_ptr, N: tl.constexpr):
    """
    Place original indices into out_ptr in globally stable sorted order.
    For each original index i (0..N-1):
      v = flat[i]
      pos = offsets[v] + rank[v]
      out[pos] = i
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # For each i, compute v, pos, and store i to out[pos]
    for i in range(N):
        v = tl.load(flat_ptr + i)
        rank_v = tl.load(rank_ptr + v)
        offset_v = tl.load(offsets_ptr + v)
        pos = offset_v + rank_v
        # write i (int32) to out[pos]
        tl.store(out_ptr + pos, i)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only forward:
        - Flatten topk_idx to a 1D int32 tensor.
        - Compute sorted_token_indices (int32 permutation of [0..N-1]) using Triton counting-sort style stable sort.
        - Compute expert_offsets (int32 of length num_experts+1) using Triton histogram and exclusive prefix sum.
        Returns:
          sorted_token_indices: int32 tensor of shape (N,)
          expert_offsets: int32 tensor of shape (num_experts + 1,)
        """
        device = topk_idx.device
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = flat.numel()
        num_experts = 256  # match original

        # 1) Compute sorted_token_indices via Triton stable counting-sort
        #    a) counts per value in [0..N-1]
        counts_val = torch.zeros(N, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_values_kernel[grid](flat, counts_val, N, BLOCK)

        #    b) exclusive prefix sum over counts_val to get offset per value
        offsets_val = torch.zeros(N, dtype=torch.int32, device=device)
        scan_exclusive_values_kernel[(N,)](counts_val, offsets_val, N)

        #    c) compute stable local rank per value using atomics
        rank = torch.zeros(N, dtype=torch.int32, device=device)
        rank_atomic_kernel[grid](flat, rank, N)

        #    d) place elements into sorted order
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        placement_kernel[grid](flat, sorted_token_indices, rank, offsets_val, N)

        # 2) Compute expert offsets via Triton histogram and exclusive prefix sum over num_experts
        counts_exp = torch.zeros(num_experts, dtype=torch.int32, device=device)
        histogram_kernel[grid](flat, counts_exp, num_experts, N, BLOCK)

        # exclusive prefix sum for expert offsets
        expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
        scan_exclusive_kernel[(num_experts,)](counts_exp, expert_offsets, num_experts)

        # final expert_offsets of length num_experts + 1, with last element as total count
        total = int(counts_exp.sum().item())
        final_expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        final_expert_offsets[:num_experts] = expert_offsets
        final_expert_offsets[num_experts] = total

        return sorted_token_indices, final_expert_offsets


def run(*args):
    return ModelNew()(*args)
