import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices_blocked(a_ptr, out_ptr, N: tl.int32, BLOCK: tl.int32):
    # Each program handles a block of indices [pid*BLOCK : (pid+1)*BLOCK]
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    idxs = start + tl.arange(0, BLOCK)  # vector of indices in this block
    mask = idxs < N  # valid indices within N

    # Load values for these indices
    # For masked positions, load 0 (will be masked out when comparing ranks).
    vi = tl.load(a_ptr + idxs, mask=mask, other=0)  # int32 vector

    # Compute rank for each index in the block (scan over all positions j)
    # We implement a simple scan: for j in 0..N-1, build vj scalar and count ranks
    # Note: this is O(N^2) per index but robust and avoids in-place hazards.
    for j in range(0, N):
        # Load vj scalar (guarded by mask for idxs); compute less and tie for each vi
        vj = tl.load(a_ptr + j, mask=(j < N), other=0)  # scalar
        less = (vj < vi)  # vector bool
        tie = (vj == vi) & (j < idxs)  # stable tie-breaker; j < idxs is vector comparison
        # Convert bools to int32 for summation
        less_i = less.to(tl.int32)
        tie_i = tie.to(tl.int32)
        rank = tl.sum(less_i + tie_i, axis=0)  # scalar rank per index in block

        # Store the original index into out at position 'rank' (stable ordering)
        # Only store for valid indices
        tl.store(out_ptr + rank, idxs, mask=mask & (j < N))


@triton.jit
def _histogram_kernel(a_ptr, N: tl.int32, histogram_ptr, num_buckets: tl.int32):
    # Each program handles one element: atomic add to its bucket
    i = tl.program_id(axis=0)
    if i < N:
        vi = tl.load(a_ptr + i)
        # Values are int32 in [0, num_buckets-1], so atomic add is safe.
        tl.atomic_add(histogram_ptr + vi, 1)


@triton.jit
def _inclusive_scan_prefix_sum(in_ptr, out_ptr, num_buckets: tl.int32):
    # Single-program inclusive scan: out[k] = sum_{i=0..k} in[i]
    acc = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_buckets):
        acc += tl.load(in_ptr + k)
        tl.store(out_ptr + k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D and ensure int32 on device
        flat = topk_idx.reshape(-1)
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)
        flat = flat.contiguous()
        N = flat.numel()
        device = flat.device
        num_experts = 256  # matches original code

        # Allocate output permutation (length N), int32 for kernel, will cast to int64 for return
        sorted_perm_i32 = torch.empty(N, dtype=torch.int32, device=device)

        # 1) Triton stable argsort kernel: compute permutation indices
        BLOCK = 128  # process 128 indices per program; adjust as needed
        grid = (triton.cdiv(N, BLOCK),)
        _stable_argsort_indices_blocked[grid](flat, sorted_perm_i32, N, BLOCK)

        # 2) Triton histogram of expert IDs: count occurrences per bucket
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_experts)

        # 3) Triton inclusive prefix sum to produce expert_offsets of length (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_experts)

        # Return sorted_token_indices (int64) and expert_offsets (int32)
        return sorted_perm_i32.to(torch.int64), offsets


def run(*args):
    return ModelNew()(*args)
