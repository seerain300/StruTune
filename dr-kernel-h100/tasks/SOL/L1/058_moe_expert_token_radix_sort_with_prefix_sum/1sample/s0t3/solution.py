import torch
import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Builds a histogram of flat indices (int32) into counts_ptr[0..255] using atomic adds.
    Each program instance processes BLOCK elements.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values; other=0 is fine for masked-out lanes
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)

    # Atomic add into the counts bin for each value
    # Note: Triton supports int32 atomic_add. We assume flat values are in [0, 255].
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_inclusive_kernel(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Computes inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[1..M].
    offsets_ptr[0] must be initialized to 0 on the host.
    This kernel assumes M is small (256) and uses a simple loop.
    """
    # We use a single program instance to do the scan; grid size is (1,)
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(M):
        acc += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized ModelNew:
        - Sorts flattened indices using torch.argsort(stable=True) to get permutation.
        - Uses Triton kernels to compute histogram of expert indices and prefix sum (offsets).
        """
        # Ensure CUDA execution
        if not topk_idx.is_cuda:
            # If input somehow arrives on CPU, move it to current CUDA device
            device = torch.device("cuda")
            topk_idx = topk_idx.to(device)

        # Flatten indices
        flat = topk_idx.reshape(-1).contiguous()  # int32
        N = flat.numel()
        device = flat.device

        # Compute stable sort permutation via PyTorch (robust and fast)
        # We sort the flattened indices (values) and return positions (indices).
        # Note: argsort on int32 will work fine; values are in [0, 255].
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # Histogram of expert indices using Triton (counts of each value in [0..255])
        counts = torch.zeros(256, dtype=torch.int32, device=device)

        # Launch histogram kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_atomic_kernel[grid](flat, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # Compute offsets (cumulative counts) using Triton inclusive scan over 256 elements
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0  # initialize first offset
        prefix_sum_inclusive_kernel[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
