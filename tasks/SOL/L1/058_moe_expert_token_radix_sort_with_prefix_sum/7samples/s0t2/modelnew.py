import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N: tl.constexpr):
    """
    Triton kernel to compute histogram (bincount) of the 1D int32 array x_ptr of length N
    into counts_ptr of length 256 (indices 0..255). We assume all x_ptr[i] are in [0, 255].
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values; masked-out lanes get 0
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    vals = vals.to(tl.int32)

    # For each lane, atomically add 1 to counts[vals] if in range
    # We guard with mask to avoid out-of-range accesses.
    # Since we assumed vals in [0, 255], this is fine.
    for i in range(0, BLOCK):
        v = vals[i]
        m = mask[i]
        if m and (v >= 0) and (v <= 255):
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of x_ptr (length L) into y_ptr (length L).
    x_ptr: int32 input
    y_ptr: int64 output
    We do this in a single program by iterating up to L steps (257 here).
    """
    acc = tl.zeros((), dtype=tl.int64)
    # Loop up to L; L is constexpr so Triton can unroll or handle it.
    for i in range(0, L):
        xi = tl.load(x_ptr + i).to(tl.int64)
        acc += xi
        tl.store(y_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # Ensure we operate on the device of the input
        device = topk_idx.device
        dtype = topk_idx.dtype  # expect int32

        # Flatten to 1D
        flat = topk_idx.reshape(-1)  # int32
        N = flat.numel()

        # Triton bincount: counts per expert id in [0, 255]
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        # Launch kernel: grid over N with BLOCK=1024
        grid = (triton.cdiv(N, 1024),)
        bincount_kernel[grid](flat, counts, N=N)

        # Allocate int64 expert_offsets of length 257
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int64, device=device)

        # Inclusive prefix sum in Triton
        inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, L=self.num_experts + 1)

        # sorted_token_indices: stable argsort of flattened indices (PyTorch)
        # This returns the permutation of [0, N-1] sorted by the values at those indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets