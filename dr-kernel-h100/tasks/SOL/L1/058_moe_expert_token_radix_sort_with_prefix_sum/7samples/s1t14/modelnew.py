import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(inp_ptr, counts_ptr, n_elements, BLOCK_SIZE: tl.constexpr, NUM_BINS: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values; cast to int32
    vals = tl.load(inp_ptr + offsets, mask=mask, other=0)
    vals = vals.to(tl.int32)

    # For each possible bin, do an atomic add for matching elements
    # NUM_BINS is constexpr (256), so Triton can unroll this loop
    for b in range(NUM_BINS):
        # Broadcast b to vector of length BLOCK_SIZE
        eq = vals == b
        # Only perform atomic add where mask and eq are true
        tl.atomic_add(counts_ptr + b, tl.sum(eq & mask, axis=0))


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_BINS: tl.constexpr):
    # Single-program inclusive scan across NUM_BINS
    # offsets_ptr is length NUM_BINS + 1
    for i in range(NUM_BINS):
        prev = offsets_ptr[i]
        cur = tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, prev + cur)
    # Set offsets[256] = N (N is not known inside kernel; we set in host code)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block_size: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.block_size = block_size

    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        topk_idx = args[0]
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels"
        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous().to(torch.int32)

        n = flat.numel()

        # 1) Triton histogram to compute counts per expert ID (length 256)
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)

        grid = (triton.cdiv(n, self.block_size),)
        _histogram_counts_kernel[grid](flat, counts, n, BLOCK_SIZE=self.block_size, NUM_BINS=self.num_experts)

        # 2) Triton inclusive prefix sum to produce offsets (length num_experts + 1)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        # Initialize offsets[0] = 0
        offsets[0] = 0
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, NUM_BINS=self.num_experts)

        # Set offsets[-1] = N (total tokens)
        offsets[-1] = n

        # 3) Stable sort indices using PyTorch for correctness
        sorted_indices = torch.sort(flat, stable=True)[1].to(torch.int32)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32)
        return sorted_indices, offsets