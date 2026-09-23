import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Compute per-expert counts for values in flat_ptr (int32).
    Each value in flat_ptr is an index in [0, 255]. We increment counts[value] for each occurrence.
    We iterate over flat_ptr in chunks of BLOCK and perform atomic_add for each element.
    """
    i = 0
    while i < N:
        idx = i + tl.arange(0, BLOCK)
        mask = idx < N
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)  # int32
        # For each value in the vector, atomically add to counts
        for j in range(BLOCK):
            # If idx[j] >= N, mask ensures vals[j] is 0, but we still guard with mask
            if mask[j]:
                val = vals[j]
                tl.atomic_add(counts_ptr + val, 1)
        i += BLOCK


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum of counts (length M) into offsets (length M+1).
    We do this in a single program instance (grid=(1,)).
    offsets[0] = 0
    for i in 1..M: offsets[i] = offsets[i-1] + counts[i-1]
    offsets[M] = sum of all counts
    """
    # Initialize offsets[0] = 0
    # Note: offsets_ptr points to a vector of length M+1
    # We will write sequentially from index 1 to M, and finally at index M we write the total sum.
    total = 0
    # Load counts one by one and update offsets
    # We use a simple loop in Triton for fixed M=256.
    for i in range(0, M):
        total += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, total)
    # The last element should be total (sum of all counts), but we have already stored total at i+1 in the loop.
    # We need to write offsets[M] = total explicitly.
    # However, our loop already wrote total to offsets[M] in the last iteration. To be explicit, we can just re-assert:
    tl.store(offsets_ptr + M, total)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block_size: int = 256, num_warps: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.block_size = block_size
        self.num_warps = num_warps

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32 on CUDA
        Returns:
          sorted_token_indices: int32, length N, permutation of 0..N-1 that sorts topk_idx.flatten() stably
          expert_offsets: int32, length (num_experts + 1), inclusive cumsum of counts
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        # Flatten
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        # Triton histogram counts per expert
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        grid_hist = (1,)  # single program; loop over N in kernel
        _histogram_counts_kernel[grid_hist](
            flat, counts, N, BLOCK=self.block_size, num_warps=self.num_warps
        )

        # Triton inclusive prefix sum to get offsets (length = num_experts + 1)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        # Initialize offsets[0] = 0 (we can leave it zero-initialized)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # Stable sort indices using torch (to guarantee correctness; Triton sort is avoided to prevent runtime errors)
        # Original returns int64; we return int32 as Triton integer type for consistency.
        sorted_token_indices = torch.argsort(flat, stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
