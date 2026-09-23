import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(x_ptr, counts_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load values as int32
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # Atomic add into counts per value
    # Note: vals are in [0, num_experts-1] by construction; other lanes masked won't add.
    for e in range(num_experts):
        # Create mask for vals == e
        m = mask & (vals == e)
        # Accumulate adds: one per lane where m is true
        tl.atomic_add(counts_ptr + e, m.to(tl.int32))


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    # Single program computes inclusive prefix sum over counts
    acc = tl.zeros((), dtype=tl.int32)
    # Store initial offset 0 at position 0 (offsets[0] = 0)
    tl.store(offsets_ptr + 0, acc)
    for e in range(num_experts):
        acc += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + 1 + e, acc)


@triton.jit
def _counting_sort_stable_kernel(flat_ptr, out_idx_ptr, global_cum_ptr, n_elements: tl.int32, num_experts: tl.int32, BLOCK_SIZE: tl.constexpr):
    # Each program handles a block of elements to assign indices. We do a simple loop over i and e.
    # Triton does not support arbitrary loops with runtime bounds well; we can process one element per program id for simplicity.
    # Alternatively, launch grid=(n,) and each program handles one i. This is acceptable for correctness.
    pid = tl.program_id(axis=0)
    # Scalar i
    i = pid
    mask_i = i < n_elements
    # Load value for this i
    val = tl.load(flat_ptr + i, mask=mask_i, other=0)
    # We need to set out_idx[global_cum[val]] = i and then increment global_cum[val]
    # We loop over experts e and perform these operations
    for e in range(num_experts):
        # If val == e, place index i at current global_cum[e] and increment global_cum[e]
        eq = (val == e)
        # Load current cumulative for this expert
        curr = tl.load(global_cum_ptr + e)
        # Compute destination position
        dest = curr
        # Store index i at that position
        tl.store(out_idx_ptr + dest, i, mask=mask_i & eq)
        # Increment global_cum[e]
        tl.store(global_cum_ptr + e, curr + 1, mask=eq)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block_size: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.block_size = block_size

    def forward(self, topk_idx: torch.Tensor):
        # Ensure we're on CUDA and int32
        assert topk_idx.is_cuda, "ModelNew expects CUDA tensors"
        flat = topk_idx.reshape(-1).contiguous()
        assert flat.dtype == torch.int32, "flat must be int32"
        n = flat.numel()
        device = flat.device

        # 1) Triton histogram of expert counts
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        grid_hist = (triton.cdiv(n, self.block_size),)
        _histogram_counts_kernel[grid_hist](flat, counts, n, self.num_experts, BLOCK_SIZE=self.block_size, num_warps=4)

        # 2) Triton inclusive prefix sum to produce expert offsets
        offsets = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # 3) Triton stable counting sort to produce sorted_token_indices
        # Allocate output permutation and global cum counts
        out_idx = torch.empty(n, dtype=torch.int32, device=device)
        global_cum = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        # Launch one program per element i
        grid_sort = (n,)
        _counting_sort_stable_kernel[grid_sort](flat, out_idx, global_cum, n, self.num_experts, BLOCK_SIZE=1, num_warps=1)

        return out_idx, offsets