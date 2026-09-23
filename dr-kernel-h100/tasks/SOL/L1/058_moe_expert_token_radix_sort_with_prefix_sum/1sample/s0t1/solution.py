import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Triton kernel that builds a histogram of indices in flat_ptr[0:N] into counts_ptr[0:256].
    Assumes indices are in [0, 255]. Each thread processes BLOCK elements and atomically
    increments the corresponding bin.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load indices; masked elements are ignored via mask
    idx = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
    # Increment counts atomically. For masked (out-of-range) positions, idx==0, but mask prevents them from being used.
    # Each index in [0,255] corresponds to a valid bin.
    # We rely on counts_ptr being int32; atomic_add works on int32.
    # Note: this is unconditionally incremented for valid positions; counts are correct afterward.
    tl.atomic_add(counts_ptr + idx, 1, mask=mask)


@triton.jit
def _prefix_sum_kernel(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Triton kernel that computes prefix sum of counts_ptr[0:M] and writes to offsets_ptr[1:M+1].
    Assumes counts_ptr is a 1D int32 vector of length M (here M=256).
    We set offsets_ptr[0] on host before launch to 0.
    """
    # Vectorize along M; Triton can handle this small dimension efficiently.
    # We perform the inclusive scan using a simple loop that is unrolled at compile time since M is constexpr.
    # But Triton doesn't have a built-in tl.cumsum across arbitrary tensors; we implement a manual scan.
    # Compute running sum and write inclusive prefix to offsets_ptr[1:].
    running = tl.zeros((), dtype=tl.int32)
    # Unrolled loop since M is constexpr
    for i in range(M):
        val = tl.load(counts_ptr + i)  # scalar int32
        running += val
        tl.store(offsets_ptr + i + 1, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - Uses Triton to build a histogram of expert indices and compute offsets (cumulative counts).
        - Keeps stable sort in PyTorch (torch.argsort), as required outputs are indices only.
        """
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        # Flatten indices
        flat = topk_idx.reshape(-1)  # int32 on GPU
        N = flat.numel()
        device = flat.device

        # Stable sort indices (original behavior): returns permutation (sorted positions)
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # Prepare histogram counts for 256 experts
        counts = torch.zeros(256, dtype=torch.int32, device=device)

        # Launch Triton histogram kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # Prepare offsets tensor and compute prefix sum via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        _prefix_sum_kernel[grid](counts, offsets, M=256)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
