import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(x_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    Build histogram of values in x_ptr (int32) into counts_ptr (int32) of length 256.
    Each element in x_ptr is an index in [0, 255]. We use atomic_add to accumulate.
    Grid is 1D; each program handles BLOCK elements.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values; masked positions get 0 (safe sentinel)
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Atomically add 1 to counts[vals] for valid positions
    # Note: vals are assumed in [0, 255]. Mask ensures we don't add for out-of-range lanes.
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def inclusive_scan_prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.int32):
    """
    Compute inclusive prefix sum over counts_ptr (length M) and write to offsets_ptr (length M+1).
    offsets_ptr[0] must be initialized to 0. We write offsets_ptr[i] = sum_{j=0..i-1} counts[j].
    This kernel runs as a single program instance over M elements.
    """
    # Loop over i from 1 to M (inclusive scan)
    # Note: Triton loops are supported; M is a constexpr-known size (256).
    for i in range(1, M):
        # Load current count
        count_i = tl.load(counts_ptr + i)
        # Sum of previous counts (scalar)
        prev_sum = 0
        for j in range(0, i):
            prev_sum += tl.load(counts_ptr + j)
        # Inclusive prefix: offsets[i] = prev_sum + count_i
        tl.store(offsets_ptr + i, prev_sum + count_i)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA tensors; Triton kernels require CUDA
        assert topk_idx.is_cuda, "Input topk_idx must be on CUDA device for Triton kernels."

        # Flatten to 1D
        flat = topk_idx.reshape(-1)  # int32 on GPU

        device = flat.device
        N = flat.numel()

        # 1) Stable sort permutation via PyTorch (robust and correct)
        # Return indices (sorted positions) for values flat.
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        histogram_atomic_kernel[grid](flat, counts, N=N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        inclusive_scan_prefix_sum_kernel[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
