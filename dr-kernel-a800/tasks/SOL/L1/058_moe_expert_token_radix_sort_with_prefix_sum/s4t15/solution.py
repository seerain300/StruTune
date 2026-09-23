import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: histogram per expert id using vector loads and scalar comparisons.
# Each program handles BLOCK elements and accumulates counts in registers, then
# does a single atomic add per expert bin.
@triton.jit
def _histogram_kernel(flat_ptr, N, counts_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a tile of flattened indices
    idx = tl.load(flat_ptr + offsets, mask=mask, other=0)
    idx = idx.to(tl.int32)

    # Accumulate counts per expert bin in registers
    for i in range(num_experts):
        matches = idx == i  # vector boolean
        cnt = tl.sum((matches & mask).to(tl.int32), axis=0)  # scalar count
        tl.atomic_add(counts_ptr + i, cnt)


# Triton kernel: compute inclusive cumulative offsets per expert using block-wise scan.
# Each program processes CHUNK experts at a time, computes inclusive scan locally, and
# writes into offsets. It also maintains a carry for prefix across chunks.
@triton.jit
def _compute_offsets_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr, CHUNK: tl.constexpr):
    carry = 0
    for chunk_start in range(0, num_experts, CHUNK):
        # Local counts for this chunk: vector of length CHUNK
        local_counts = tl.zeros((CHUNK,), dtype=tl.int32)
        # Load counts for this chunk
        for j in range(CHUNK):
            i = chunk_start + j
            # Load count for expert i; if i >= num_experts, local_counts[j] stays 0
            cnt = tl.load(counts_ptr + i)
            local_counts[j] = cnt
        # Inclusive scan within the chunk: sequential pairwise update (CHUNK is small)
        running = 0
        for j in range(CHUNK):
            running += local_counts[j]
            i_abs = chunk_start + j
            # Only write if within range
            if i_abs < num_experts:
                tl.store(offsets_ptr + i_abs + 1, carry + running)
        # Update carry with the sum of this chunk's counts
        chunk_sum = 0
        for j in range(CHUNK):
            i = chunk_start + j
            if i < num_experts:
                chunk_sum += local_counts[j]
        carry += chunk_sum


def _run_triton_only(topk_idx: torch.Tensor):
    # Ensure CUDA and contiguous
    if not topk_idx.is_cuda:
        topk_idx = topk_idx.to(device="cuda")
    topk_idx = topk_idx.contiguous()

    # Flatten
    flat = topk_idx.reshape(-1)
    N = flat.numel()

    # Histogram counts (per expert)
    num_experts = 256
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

    # Triton histogram kernel: process BLOCK elements per program
    BLOCK = 4096  # tuned for typical sizes; Triton will compile this
    grid = (triton.cdiv(N, BLOCK),)
    _histogram_kernel[grid](flat, N, counts, num_experts=num_experts, BLOCK=BLOCK, num_warps=8, num_stages=2)

    # Compute expert offsets via Triton block-wise scan
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    CHUNK = 64  # process 64 experts per program; fine for num_experts=256
    _compute_offsets_kernel[(1,)](counts, offsets, num_experts=num_experts, CHUNK=CHUNK)

    # Stable sort of flattened indices in PyTorch (data-independent on num_experts)
    sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

    return sorted_token_indices, offsets


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; required by the evaluation harness.

    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]
        # Triton-only execution
        return _run_triton_only(topk_idx)


def run(*args):
    return ModelNew()(*args)
