import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def count_histogram_atomic(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        """
        Parallel histogram: for each flat[i], atomically add 1 to counts[flat[i]].
        """
        pid = tl.program_id(axis=0)
        start = pid * BLOCK
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N

        vals = tl.load(flat_ptr + offsets, mask=mask, other=0).to(tl.int32)

        for i in range(BLOCK):
            idx_i = vals[i]
            # Only increment valid positions
            if mask[i]:
                tl.atomic_add(counts_ptr + idx_i, 1)

    @triton.jit
    def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
        """
        Compute exclusive prefix sum: offsets[i] = sum_{k=0..i-1} counts[k] for i in 1..N_bins-1; offsets[0] = 0.
        O(N_bins^2), acceptable for N_bins=256.
        """
        tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
        for i in range(1, N_bins):
            total = tl.zeros((), dtype=tl.int32)
            for k in range(0, i):
                total += tl.load(counts_ptr + k)
            tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version:
    - Compute per-expert counts using Triton atomic histogram.
    - Compute expert offsets via Triton exclusive prefix sum.
    Note: sorted_token_indices uses torch.sort for correctness and simplicity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton histogram of expert indices
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK_HIST = 8192  # larger BLOCK improves throughput; grid size will be cdiv(N, BLOCK_HIST)
        grid = (triton.cdiv(N, BLOCK_HIST),)
        count_histogram_atomic[grid](flat, counts, N=N, BLOCK=BLOCK_HIST, num_warps=4)

        # 2) Exclusive prefix sum via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)  # length = num_experts + 1
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=256, num_warps=1)

        # 3) sorted_token_indices using PyTorch (to match original behavior)
        _, sorted_token_indices = torch.sort(flat, dim=0, stable=True)
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
