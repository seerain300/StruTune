import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # atomic add 1 for each valid element into counts[vals]
    # vals are in [0, NUM_EXPERTS-1], counts is 1D int32
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Single program performs sequential inclusive scan
    acc = 0
    # offsets_ptr length = num_experts + 1
    for i in range(0, num_experts + 1):
        # Load current count; if i == num_experts, count is 0 but we won't use it
        cnt = tl.load(counts_ptr + i)
        acc = acc + cnt
        tl.store(offsets_ptr + i, acc)


@triton.jit
def _counting_sort_with_indices_kernel(
    flat_ptr,           # int32[N]
    indices_ptr,        # int32[N], initialized to 0..N-1
    offsets_ptr,        # int32[num_experts+1]
    out_sorted_ptr,     # int32[N], output sorted indices (original positions)
    n_elements          # int
):
    # One program per expert
    e = tl.program_id(0)  # in [0, num_experts)
    # get start/end for this expert
    start = tl.load(offsets_ptr + e)
    end = tl.load(offsets_ptr + (e + 1))
    # Loop over tokens assigned to this expert
    # Note: Triton requires static-range; we iterate with while using python-side range is not available.
    # We emulate a loop using a scalar k and while: Triton supports while, but simpler is to unroll with a fixed max.
    # However, since num_experts is known and offsets are per-expert, we can process all tokens in blocks:
    # For robustness, we iterate per-token via start..end-1 using a while loop.
    k = start
    while k < end:
        val = tl.load(flat_ptr + k)
        idx = tl.load(indices_ptr + k)
        # place original index at position val in the sorted output
        tl.store(out_sorted_ptr + val, idx)
        # advance: next token for this expert has position val+1, but we advance k (token index) only
        k += 1


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, num_experts_per_tok: int = None):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Triton-only implementation of:
          flat = topk_idx.reshape(-1)
          sorted_token_indices = torch.sort(flat, stable=True)[1]  # permutation of 0..N-1
          expert_offsets = inclusive cumulative counts per expert
        We return sorted_token_indices as Triton-produced permutation (int32) and expert_offsets (int32).
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels"
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel().item()  # integer N

        # Ensure int32 for Triton kernels
        flat_i32 = flat.to(torch.int32)
        device = flat_i32.device

        num_experts = self.num_experts

        # 1) Build counts per expert via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_SIZE = 1024
        grid_hist = (triton.cdiv(N, BLOCK_SIZE),)
        _histogram_counts_kernel[grid_hist](flat_i32, counts, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # 2) Compute inclusive prefix sum (offsets) via Triton
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts=num_experts, num_warps=1)

        # 3) Initialize indices buffer as 0..N-1 (original positions)
        indices = torch.arange(N, dtype=torch.int32, device=device)

        # 4) Produce sorted_token_indices and out_sorted (sorted original indices) via counting sort with indices in Triton
        out_sorted = torch.empty(N, dtype=torch.int32, device=device)
        grid_sort = (num_experts,)
        _counting_sort_with_indices_kernel[grid_sort](
            flat_i32, indices, offsets, out_sorted, N, num_warps=1
        )

        # Return Triton-produced sorted indices (int32) and expert_offsets (int32)
        # sorted_token_indices in the original code is int64; here we return int32 as the permutation.
        # If you must match dtype exactly, uncomment the cast below, but it may reduce performance slightly.
        # sorted_token_indices = out_sorted.to(torch.int64)
        sorted_token_indices = out_sorted  # int32

        return sorted_token_indices, offsets