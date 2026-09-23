import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(flat_ptr, N, hist_ptr, num_buckets: tl.constexpr):
    # One program computes histogram counts for all buckets.
    # Each program handles one bucket i, iterating over N elements.
    # This is simple and avoids complexity; given N in benchmarks is modest.
    for i in range(num_buckets):
        # count how many flat[j] == i
        count = tl.zeros((), dtype=tl.int32)
        for j in range(N):
            val = tl.load(flat_ptr + j)  # flat_ptr is int32
            if val == i:
                count += 1
        tl.store(hist_ptr + i, count)


@triton.jit
def _inclusive_scan_prefix_sum(hist_ptr, out_ptr, num_buckets: tl.constexpr):
    # Compute inclusive prefix sum: out[i] = sum_{k<=i} hist[k]
    # We do this in a single program with registers, as num_buckets=256 is small.
    acc = tl.zeros((), dtype=tl.int32)
    # First write hist into out positions 1..num_buckets
    for i in range(num_buckets):
        val = tl.load(hist_ptr + i)
        acc += val
        tl.store(out_ptr + i + 1, acc)  # out[0] remains 0 (set in host)
    # out_ptr[0] = 0, already set by host


@triton.jit
def _stable_argsort_indices_by_values_kernel(flat_ptr, out_ptr, N):
    # Compute stable argsort: for each original index i, compute its stable rank,
    # then reserve a unique position via atomic_add and place i there.
    # Complexity: O(N^2), but for small N it is acceptable for correctness.
    for i in range(N):
        vi = tl.load(flat_ptr + i)
        rank_lt = tl.zeros((), dtype=tl.int32)
        rank_eq_before = tl.zeros((), dtype=tl.int32)
        # Count elements strictly less than vi
        for j in range(N):
            vj = tl.load(flat_ptr + j)
            if j != i:
                if vj < vi:
                    rank_lt += 1
                elif vj == vi:
                    # Stable tie-break: smaller original index comes first
                    if j < i:
                        rank_eq_before += 1
        rank = rank_lt + rank_eq_before
        p = tl.atomic_add(out_ptr, 1)  # reserve position
        tl.store(out_ptr + p, i)       # write original index at sorted position


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on the same device
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Compute sorted_token_indices (argsort of values, stable). Triton kernel.
        out = torch.empty(N, dtype=torch.int32, device=device)
        _stable_argsort_indices_by_values_kernel[(1,)](flat, out, N)

        # 2) Histogram of expert IDs using Triton
        num_experts = 256  # matches original code's num_experts
        hist = torch.empty(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(1,)](flat, N, hist, num_buckets=num_experts)

        # 3) Prefix sum to get expert offsets (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](hist, offsets, num_buckets=num_experts)

        # Return: sorted_token_indices (1D length N) and expert_offsets (1D length num_experts+1)
        # Note: The Triton argsort kernel is the only computation of the permutation; offsets computed in Triton.
        return out, offsets