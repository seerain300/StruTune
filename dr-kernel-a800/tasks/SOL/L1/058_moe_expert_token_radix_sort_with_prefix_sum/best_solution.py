import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(flat_ptr, N, counts_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Triton histogram kernel:
    - flat_ptr: pointer to int32 flattened indices (length N).
    - N: total number of elements (runtime).
    - counts_ptr: pointer to int32 counts of length num_experts.
    - num_experts: number of experts (compile-time for loop).
    - BLOCK: number of elements processed per program (compile-time).
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load a block of indices; for out-of-bounds, set to 0 (won't affect counts due to mask)
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)

    # For each expert bin, compute number of matches in this block and atomic add to global counts
    for e in range(num_experts):
        # matches: True where vals == e, but only for valid offs
        matches = (vals == e) & mask
        # Reduce matches to int32 count
        cnt = tl.sum(matches.to(tl.int32), axis=0)
        # Atomically accumulate into global counts[e]
        tl.atomic_add(counts_ptr + e, cnt)


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Triton inclusive prefix sum kernel:
    - counts_ptr: pointer to int32 counts of length num_experts.
    - offsets_ptr: pointer to int32 offsets of length num_experts + 1.
    - num_experts: number of experts (compile-time).
    - Computes: offsets[0] = 0; offsets[i+1] = sum_{j=0..i} counts[j].
    This kernel is small and runs as a single program; it reads counts and writes offsets.
    """
    # We'll do a simple loop; Triton allows loops with tl.constexpr bounds.
    running = 0
    # offsets_ptr[0] = 0 explicitly before this kernel; we write from 1..num_experts
    for i in range(num_experts):
        running += tl.load(counts_ptr + i)
        # Store inclusive sum at i+1
        tl.store(offsets_ptr + i + 1, running)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Forward flattens topk_idx, computes per-expert counts in Triton, and expert_offsets via Triton prefix sum.
    - Sorting remains in PyTorch (GPU), as it's independent of num_experts and already efficient.
    Entry point: ModelNew
    """

    def __init__(self):
        super().__init__()
        # No parameters; required by the evaluation harness (NewModule).
        self.num_experts = 256  # fixed as in the original code
        # You can tune BLOCK; 1024 is a good default for these sizes.
        self.BLOCK = 1024
        self.num_warps = 4  # suitable for BLOCK=1024

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32, on CUDA.
        Returns:
          sorted_token_indices: torch.Tensor of int32, shape (N,), sorted stable permutation.
          expert_offsets: torch.Tensor of int32, shape (num_experts + 1,), inclusive prefix sums.
        """
        if not topk_idx.is_cuda:
            # Ensure GPU execution; the evaluation environment provides CUDA device.
            raise RuntimeError("topk_idx must be on CUDA device")

        # Ensure contiguous
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.view(-1)  # shape (N,)
        N = flat.numel()

        # 1) Triton histogram to get counts per expert id
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        grid = (triton.cdiv(N, self.BLOCK),)
        _histogram_kernel[grid](flat, N, counts, num_experts=self.num_experts, BLOCK=self.BLOCK, num_warps=self.num_warps)

        # 2) Triton inclusive prefix sum to produce expert_offsets
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        # Initialize expert_offsets[0] = 0
        expert_offsets[0] = 0
        _inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, num_experts=self.num_experts, num_warps=1)

        # 3) Stable sort of flattened indices (PyTorch, data-independent on num_experts)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
