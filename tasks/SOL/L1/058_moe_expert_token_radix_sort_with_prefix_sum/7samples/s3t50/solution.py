import torch
import triton
import triton.language as tl


@triton.jit
def _stable_rank_argsort_indices(flat_ptr, N, out_ptr):
    # One program per original index i
    i = tl.program_id(0)  # scalar int32
    # Compute rank by scanning all j
    rank = tl.zeros((), dtype=tl.int32)
    # N is scalar int32 passed in; j is scalar int32
    for j in range(0, N):
        a_i = tl.load(flat_ptr + i)
        a_j = tl.load(flat_ptr + j)
        less = a_j < a_i
        tie = a_j == a_i
        # Stable tie-breaking: ensure smaller j comes first
        tie_and_order = tie & (j < i)
        rank += less + tie_and_order
    # Write i to out[rank]; out is int32
    tl.store(out_ptr + rank, i)


@triton.jit
def _histogram_kernel(flat_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    # One program per element; atomically add into the corresponding bucket
    i = tl.program_id(0)  # scalar int32
    # Guard against i >= N (launch grid should be N, but keep mask for safety)
    if i < N:
        val = tl.load(flat_ptr + i)
        # val is int32 in [0, num_buckets-1]; no need for masking
        # atomic add: histogram[val] += 1
        tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(hist_ptr, offsets_ptr, num_buckets: tl.constexpr):
    # Single-program inclusive scan across num_buckets
    running = tl.zeros((), dtype=tl.int32)
    # Initialize offsets[0] = 0 (host sets this), then compute inclusive sums
    for k in range(0, num_buckets):
        running += tl.load(hist_ptr + k)
        tl.store(offsets_ptr + k + 1, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure int32 for Triton
        flat = topk_idx.contiguous().view(-1)
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)
        N = flat.numel()
        device = flat.device

        # 1) Stable argsort via Triton: out[i] = original index of the i-th smallest value
        out = torch.empty(N, dtype=torch.int32, device=device)
        # Launch one program per i
        grid_argsort = (N,)
        _stable_rank_argsort_indices[grid_argsort](flat, N, out)

        # 2) Triton histogram of expert IDs
        num_experts = 256  # matches original code
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_buckets=num_experts)

        # 3) Triton inclusive prefix sum to produce expert_offsets (length num_experts + 1)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        # Return sorted_token_indices as int64 (original uses int64), and expert_offsets as int32
        # sorted_token_indices is out (permutation), original returns int64
        sorted_perm = out  # int32; convert to int64 as expected
        return sorted_perm.to(torch.int64), offsets


def run(*args):
    return ModelNew()(*args)
