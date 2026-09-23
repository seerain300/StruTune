import torch
import triton
import triton.language as tl


# Kernel 1: per-token histogram with atomic adds
@triton.jit
def count_experts_kernel(topk_idx_ptr, counts_ptr, N: tl.constexpr):
    """
    For each token index i in [0, N), read topk_idx_ptr[i], and atomically
    increment counts[topk_idx[i]] by 1.
    """
    pid = tl.program_id(0)
    # Each program handles one token. We assume grid = (N,)
    if pid < N:
        # Load expert ID for this token
        val = tl.load(topk_idx_ptr + pid)
        # Atomic add into counts[val]
        tl.atomic_add(counts_ptr + val, 1)


# Kernel 2: inclusive prefix sum of a small vector (num_experts = 256)
# We implement a simple loop-based scan per element. Triton will unroll with static_range.
@triton.jit
def prefix_sum_inclusive_kernel(counts_ptr, out_ptr, M: tl.constexpr):
    """
    Compute inclusive prefix sum of counts[0:M] and write to out[0:M].
    We assume M is small (e.g., 256), and use a simple scan loop.
    """
    # Single program handles the entire vector. We write out results to out_ptr[i].
    # We need a running sum across the vector elements.
    running = 0
    for i in range(0, M):
        # Load current count
        val = tl.load(counts_ptr + i)
        running += val
        tl.store(out_ptr + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Produces sorted_token_indices: permutation of 0..N-1 sorting by expert IDs.
        - Produces expert_offsets: cumulative counts per expert (num_experts+1).
        We compute the histogram via Triton; sorting remains in PyTorch for correctness.
        """
        assert topk_idx.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        # Flatten to 1D and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # Number of experts is fixed as 256 in the original code
        num_experts = 256

        # 1) Triton kernel to compute per-expert counts (histogram)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        # Launch one program per token
        grid = (N,)
        count_experts_kernel[grid](flat, counts, N=N)

        # 2) Compute expert_offsets via inclusive prefix sum
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        # We'll compute prefix sum of counts into out[1:], then set out[0] = 0
        # For small M=256, a single program with a loop is fine.
        prefix_sum_inclusive_kernel[(1,)](counts, expert_offsets[1:], M=num_experts)

        # 3) Sorting is left to PyTorch (stable=True). This is necessary to reproduce behavior.
        #    flat is a permutation of token indices; sorting by expert IDs yields the required stable order.
        sorted_indices = flat.sort(stable=True).indices

        return sorted_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
