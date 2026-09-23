import torch
import triton
import triton.language as tl


@triton.jit
def count_kernel(topk_ptr, histogram_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    For each element e in [0, N), increment histogram[topk[e]] by 1 using atomic add.
    histogram has length num_experts = 256; topk_ptr points to int32 values in [0, 255].
    """
    e = tl.program_id(0)  # each program handles one element
    # Load index; if e >= N, guard is unnecessary since grid will be set to N.
    idx = tl.load(topk_ptr + e)
    # Atomic add into histogram[idx]; assume idx in [0, 255]
    tl.atomic_add(histogram_ptr + idx, 1)


@triton.jit
def prefix_sum_kernel(histogram_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of histogram of length num_experts into offsets[0..num_experts-1].
    offsets_ptr[0] should be initialized to 0. Each program computes offsets[i] = sum_{j < i} histogram[j].
    We iterate over i in [0, num_experts-1] using a loop. The grid is (1,) since we do a single pass.
    """
    # This kernel runs as a single program with a simple loop over num_experts.
    # We pass num_experts as tl.constexpr so Triton can unroll the loop.
    # Initialize current sum to 0
    curr = tl.zeros((), dtype=tl.int32)
    # Loop over i from 0 to num_experts-1
    for i in range(0, num_experts):
        curr += tl.load(histogram_ptr + i)
        tl.store(offsets_ptr + i, curr)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run:
        - Uses Triton to build a histogram of expert indices via atomic adds.
        - Uses torch.cumsum for prefix sums (fast and simple).
        - Keeps torch.argsort for stable sorting of flattened indices.
        Returns:
          sorted_token_indices: token indices sorted by expert (int32, length N)
          expert_offsets:       cumulative offsets per expert (int32, length num_experts+1)
        """
        # Ensure we are on CUDA
        if not topk_idx.is_cuda:
            # Move to CUDA if available
            topk_idx = topk_idx.to('cuda')
        # Ensure contiguous and int32
        topk_idx = topk_idx.contiguous()
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        # Flatten
        N = topk_idx.numel()
        # Allocate histogram of length 256
        histogram = torch.zeros(256, dtype=torch.int32, device=topk_idx.device)

        # Launch Triton kernel to count occurrences
        grid = (N,)
        count_kernel[grid](topk_idx, histogram, N=N, BLOCK=1)

        # Compute prefix sums for offsets; set offsets[0] = 0 and offsets[1:] = cumsum(histogram)
        # We do this with PyTorch for simplicity and speed (num_experts is fixed and small).
        offsets = torch.cumsum(histogram, dim=0).to(torch.int32)
        # Since original code sets offsets[1:] = cumsum(...), we keep offsets[0] = 0 (torch.cumsum does this).
        # But to be explicit:
        # offsets = torch.zeros(257, dtype=torch.int32, device=topk_idx.device)
        # offsets[1:] = torch.cumsum(histogram, dim=0)

        # For clarity, construct offsets as described:
        offsets = torch.zeros(257, dtype=torch.int32, device=topk_idx.device)
        offsets[1:] = offsets[1:] + torch.cumsum(histogram, dim=0)

        # Sorting: use torch.argsort on the flattened indices to match stable=True behavior.
        # Note: The original sorts "values" (which are indices) and returns indices (sorted positions).
        flat = topk_idx.reshape(-1)
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
