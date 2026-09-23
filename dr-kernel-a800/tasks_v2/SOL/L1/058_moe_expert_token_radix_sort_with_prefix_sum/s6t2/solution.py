import torch
import triton
import triton.language as tl


@triton.jit
def count_values_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Count occurrences of each value v in [0..255] in flat_ptr (int32).
    counts_ptr is int32 of length 256, zero-initialized by host.
    Processes BLOCK elements per program and atomically adds local counts to global counts.
    Assumes values are within [0, 255] to cover all bins.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # For each v in 0..255, count occurrences in this block
    for v in range(256):
        is_v = vals == v
        count_v = tl.sum(is_v.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + v, count_v)


@triton.jit
def scan_exclusive_values_kernel(counts_ptr, offsets_ptr, N: tl.constexpr):
    """
    Exclusive prefix sum over counts_ptr (length N=256) and write to offsets_ptr (length 256).
    offset[v] = sum(counts[:v]) for v in 0..255.
    """
    start = tl.zeros((), dtype=tl.int32)
    v = 0
    while v < N:
        cnt = tl.load(counts_ptr + v)
        tl.store(offsets_ptr + v, start)
        start += cnt
        v += 1


@triton.jit
def rank_atomic_kernel(flat_ptr, rank_ptr, N, BLOCK: tl.constexpr):
    """
    Compute stable local rank for each value v in [0..255]: rank[v] = number of elements with value v and index < current index.
    Iterate over indices in blocks; for each i, read value v=flat[i] and atomically increment rank[v].
    rank_ptr is int32 of length 256, zero-initialized by host.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Loop over all positions i and atomically increment rank[flat[i]]
    for i in range(N):
        val_i = tl.load(flat_ptr + i)  # scalar load; assume val_i in [0..255]
        tl.atomic_add(rank_ptr + val_i, 1)


@triton.jit
def placement_kernel(flat_ptr, out_ptr, rank_ptr, offsets_ptr, N, BLOCK: tl.constexpr):
    """
    Place original indices into out_ptr in globally stable sorted order.
    For each original index i (0..N-1):
      v = flat[i]
      pos = offsets[v] + rank[v]
      out[pos] = i
    Assumes v in [0..255].
    """
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    for i in range(N):
        v = tl.load(flat_ptr + i)
        rank_v = tl.load(rank_ptr + v)
        offset_v = tl.load(offsets_ptr + v)
        pos = offset_v + rank_v
        tl.store(out_ptr + pos, i)


@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, num_experts: tl.constexpr, N, BLOCK: tl.constexpr):
    """
    Count occurrences of each expert id (0..num_experts-1) in orig_ptr (int32).
    counts_ptr is int32 of length num_experts, zero-initialized by host.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(orig_ptr + offs, mask=mask, other=0)
    for e in range(num_experts):
        is_e = vals == e
        count_e = tl.sum(is_e.to(tl.int32), axis=0)
        tl.atomic_add(counts_ptr + e, count_e)


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
def add_total_kernel(counts_ptr, out_ptr, num_experts: tl.constexpr):
    """
    Compute total count = sum(counts_ptr[:num_experts]) and write to out_ptr[0].
    """
    total = tl.zeros((), dtype=tl.int32)
    for e in range(num_experts):
        total += tl.load(counts_ptr + e)
    tl.store(out_ptr, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized forward:
        - Produces sorted_token_indices via Triton counting sort for values in [0..255].
        - Produces expert_offsets via Triton histogram + exclusive prefix sum.
        Returns:
          sorted_token_indices: int32 tensor of shape (N,) representing permutation that would sort the flattened tensor stably for values in [0..255].
          expert_offsets: int32 tensor of shape (num_experts + 1,) with offsets[e] = inclusive sum of counts for ids < e.
        """
        device = topk_idx.device
        orig = topk_idx.reshape(-1).contiguous()
        N = orig.numel()

        # 1) Triton counting sort for values in [0..255]
        counts_val = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_values_kernel[grid](orig, counts_val, N, BLOCK)

        # 2) Triton exclusive prefix sum of counts_val to produce offsets_val
        offsets_val = torch.zeros(256, dtype=torch.int32, device=device)
        scan_exclusive_values_kernel[(256,)](counts_val, offsets_val, 256)

        # 3) Triton stable rank via atomic increments over orig (values in [0..255])
        rank_val = torch.zeros(256, dtype=torch.int32, device=device)
        rank_atomic_kernel[grid](orig, rank_val, N, BLOCK)

        # 4) Triton placement to produce sorted_token_indices
        out = torch.empty(N, dtype=torch.int32, device=device)
        placement_kernel[grid](orig, out, rank_val, offsets_val, N, BLOCK)

        sorted_token_indices = out

        # 5) Triton histogram of original topk_idx values to compute expert offsets
        num_experts = 256  # match original
        counts_exp = torch.zeros(num_experts, dtype=torch.int32, device=device)

        # 6) Triton histogram of expert ids
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_kernel[grid_hist](orig, counts_exp, num_experts, N, BLOCK)

        # 7) Triton exclusive prefix sum for expert offsets
        offsets_exp = torch.zeros(num_experts, dtype=torch.int32, device=device)
        scan_exclusive_kernel[(num_experts,)](counts_exp, offsets_exp, num_experts)

        # 8) Triton: compute total count and write to final buffer
        total_buf = torch.empty(1, dtype=torch.int32, device=device)
        add_total_kernel[(num_experts,)](counts_exp, total_buf, num_experts)
        total = int(total_buf.item())

        final_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        final_offsets[:num_experts] = offsets_exp
        final_offsets[num_experts] = total

        return sorted_token_indices, final_offsets


def run(*args):
    return ModelNew()(*args)
