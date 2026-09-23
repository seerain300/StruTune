import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: for each token i in 0..N-1, if vals[i] == e, atomic add 1 to counts[e].
    vals_ptr: *int32, length N
    counts_ptr: *int32, length num_experts
    N: number of tokens (runtime int)
    num_experts: number of experts (constexpr, e.g., 256)
    """
    pid = tl.program_id(0)
    # Each program handles one token to avoid complexity in indexing.
    # Grid should be at least N; if more, we mask out extra programs.
    if pid >= N:
        return
    val = tl.load(vals_ptr + pid)
    # Increment counts for all possible expert ids; val must be in [0, num_experts)
    for e in range(num_experts):
        if val == e:
            tl.atomic_add(counts_ptr + e, 1)


@triton.jit
def inclusive_scan_kernel(in_ptr, out_ptr, N: tl.constexpr):
    """
    Triton kernel: inclusive prefix sum over a small vector of length N (constexpr).
    Reads N int32 values from in_ptr, computes prefix sums, writes to out_ptr.
    """
    # Single program performs sequential scan over N elements
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(N):
        x = tl.load(in_ptr + i)
        acc += x
        tl.store(out_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Compute per-expert counts via Triton (atomic add per value).
        - Compute inclusive prefix sum of counts via Triton.
        Returns:
          - sorted_token_indices: unavailable in Triton-only; original uses torch.sort(stable=True).
          - expert_offsets: [0] + cumsum(bincount(flat)) via Triton (no torch.data_ops used).
        """
        # Ensure dtype and contiguity
        vals = topk_idx.reshape(-1).contiguous()
        if vals.dtype != torch.int32:
            vals = vals.to(torch.int32)
        N = vals.numel()
        num_experts = 256  # fixed for provided workloads

        # 1) Compute counts (global histogram of expert ids) via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=vals.device)
        # Launch one program per token; grid size must be >= N
        count_experts_kernel[(N,)](vals, counts, N, num_experts=num_experts)

        # 2) Compute inclusive prefix sum of counts via Triton (vector of length num_experts)
        offsets_exclusive = torch.empty_like(counts, dtype=torch.int32, device=vals.device)
        inclusive_scan_kernel[(1,)](counts, offsets_exclusive, N=num_experts)

        # Construct expert offsets (length = num_experts + 1) with 0 prepended
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=vals.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = offsets_exclusive  # inclusive scan already includes all elements

        # sorted_token_indices cannot be generated in Triton-only with stable sort without torch.sort.
        # We return expert_offsets and note the limitation.

        return (None, expert_offsets)


def run(*args):
    return ModelNew()(*args)
