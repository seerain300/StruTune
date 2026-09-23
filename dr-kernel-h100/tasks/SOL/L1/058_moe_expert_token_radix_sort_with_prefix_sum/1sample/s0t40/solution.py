import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Compute histogram of values in flat (int32), with counts in [0..255].
    counts_ptr: int32[256], flat_ptr: int32[N].
    """
    idx = tl.arange(0, BLOCK)
    # We'll process the entire array in chunks of size BLOCK.
    # Each program handles a chunk and atomically adds 1 for each valid element.
    # Use a loop over chunks.
    for start in range(0, N, BLOCK):
        offs = start + idx
        mask = offs < N
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # atomic add for valid lanes
        tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0..M-1] into offsets_ptr[0..M], with offsets[0]=0.
    M is 256 (num_experts).
    """
    # offsets_ptr is length M+1 (int32)
    # We'll do an iterative doubling scan in-place on offsets_ptr[1..].
    # Initialize offsets[0] must be 0; caller sets it.
    # Iterative scan for M elements.
    i = 0
    running = 0
    # We use a simple while loop; Triton allows while with constexpr M.
    while i < M:
        # offsets[i] += running
        # running = offsets[i]
        # Using atomic to update shared offsets safely
        tl.atomic_add(offsets_ptr + i, running)
        running = tl.load(offsets_ptr + i)
        i += 1
        # Now add contributions from previous positions:
        # For each j in 1,2,4,8,... up to M:
        j = 1
        while j < i:
            tl.atomic_add(offsets_ptr + i, running)
            running = tl.load(offsets_ptr + i)
            j <<= 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: (batch_size, seq_len, num_experts_per_tok), int32, CUDA
        flat = topk_idx.reshape(-1)  # int32 1D tensor
        N = flat.numel()

        # 1) Stable sort via PyTorch to ensure correctness
        # Important: convert to long to match original run behavior on indices
        sorted_token_indices = torch.argsort(flat.long(), stable=True).to(torch.int32)

        # 2) Histogram via Triton: counts[0..255]
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid](flat, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton inclusive scan over 256 elements
        offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
