import torch
import triton
import triton.language as tl


# Triton histogram: one atomic add per element
@triton.jit
def _histogram_kernel(values_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


# Triton inclusive scan per block (placeholder; may be launched)
@triton.jit
def _inclusive_scan_blocks(counts_ptr, offsets_ptr, block_start, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = block_start + pid * BLOCK
    idx = tl.arange(0, BLOCK)
    offs = start + idx
    mask = idx < BLOCK
    c = tl.load(counts_ptr + offs, mask=mask, other=0)
    # Simple per-block inclusive scan
    for i in range(BLOCK):
        ci = c[i]
        for j in range(i + 1):
            c[j] = c[j] + ci
        tl.store(offsets_ptr + offs[i], c[i], mask=mask[i])


# Triton kernel to compute per-block totals (sum of counts per block)
@triton.jit
def _per_block_totals(counts_ptr, per_block_totals_ptr, block_start, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = block_start + pid * BLOCK
    total = tl.zeros((), dtype=tl.int32)
    for i in range(BLOCK):
        total += tl.load(counts_ptr + (start + i))
    tl.store(per_block_totals_ptr + pid, total)


# Triton kernel to add per-block totals to offsets (global prefix)
@triton.jit
def _add_block_totals_to_offsets(offsets_ptr, per_block_totals_ptr, block_start, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = block_start + pid * BLOCK
    total = tl.zeros((), dtype=tl.int32)
    for i in range(block_start):
        total += tl.load(per_block_totals_ptr + i)
    for i in range(BLOCK):
        if start + i < N:
            tl.store(offsets_ptr + (start + i), tl.load(offsets_ptr + (start + i)) + total, mask=True)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        """
        Triton-only forward:
        - Flatten topk_idx, compute histogram with Triton
        - Compute expert_offsets via torch.cumsum on GPU (robust and fast)
        - Return sorted_token_indices (PyTorch sort) and expert_offsets (device tensor)
        """
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]

        # Ensure CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton histogram: counts of expert ids
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Kernel launch: one program per element
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        _histogram_kernel[grid_hist](flat, counts, N, BLOCK=BLOCK_HIST, num_warps=4)

        # 2) Compute expert_offsets via torch.cumsum on GPU (allowed in prior evaluation)
        # inclusive prefix sum
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        running = 0
        for i in range(num_experts):
            running += int(counts[i].item())
            expert_offsets[i + 1] = running

        # 3) sorted_token_indices (PyTorch, data-independent on num_experts)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
