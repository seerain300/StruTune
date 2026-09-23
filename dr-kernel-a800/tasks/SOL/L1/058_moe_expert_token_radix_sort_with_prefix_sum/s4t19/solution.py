import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: per-element atomic histogram of expert indices
@triton.jit
def histogram_kernel(inp_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load flattened indices; out-of-bounds lanes will be ignored via mask
    idx = tl.load(inp_ptr + offs, mask=mask, other=0)  # int32 indices
    # Increment counts for valid lanes
    # Note: counts_ptr is a 1D array of int32, length = num_experts
    # Each valid idx in [0, 255] points to a valid count element.
    tl.atomic_add(counts_ptr + idx, 1, mask=mask)

# Triton kernel: inclusive prefix sum per-expert offsets (2D tiles)
@triton.jit
def prefix_offsets_kernel(counts_ptr, out_ptr, num_experts, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    local = tl.arange(0, BLOCK)  # vector lane indices
    idx = start + local
    mask = idx < num_experts

    # Load counts for this chunk (masked)
    vals = tl.load(counts_ptr + idx, mask=mask, other=0)  # int32

    # Compute exclusive prefix sum for this chunk
    # We'll compute carry (sum of previous chunk) separately and add to final
    carry = 0
    # First, compute exclusive scan within this block
    exclusive = tl.zeros([BLOCK], dtype=tl.int32)
    # For i in 0..BLOCK-1, set exclusive[i] = sum(vals[:i]) using scalar loop
    for i in range(BLOCK):
        # Only operate when lane i is valid
        if mask[i]:
            # sum of vals before i: we can't vectorize, so loop scalarly
            s = 0
            for j in range(i):
                if mask[j]:
                    s += vals[j]
            exclusive[i] = s

    # Now compute inclusive scan for this block: inclusive[i] = exclusive[i] + vals[i]
    inclusive_block = exclusive + vals  # vals[i] is 0 for invalid lanes

    # For valid lanes, write:
    # out[start + i] = carry + inclusive_block[i]
    for i in range(BLOCK):
        if mask[i]:
            out_offset = start + i
            tl.store(out_ptr + out_offset, carry + inclusive_block[i])

    # Update carry: sum of this block (including invalid lanes)
    # Sum only valid lanes
    chunk_sum = 0
    for j in range(BLOCK):
        if mask[j]:
            chunk_sum += vals[j]
    carry += chunk_sum

# ModelNew entry point
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; Triton kernels handle computation.

    def forward(self, *args):
        # Expect a single input tensor topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects exactly one argument: topk_idx")
        topk_idx = args[0]

        # Ensure CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten and cast to int32
        flat = topk_idx.view(-1)
        # Triton int32
        flat_i32 = flat.to(torch.int32)
        N = flat_i32.numel()

        num_experts = 256

        # 1) Triton histogram: counts of expert indices
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat_i32.device)

        # Use simple grid: one program per element; BLOCK can be 1 for robustness
        BLOCK = 1
        grid = (N,)
        histogram_kernel[grid](flat_i32, counts, N=N, BLOCK=BLOCK, num_warps=1)

        # 2) Triton inclusive prefix sum for expert_offsets (num_experts + 1)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat_i32.device)
        # We can set first element to 0 explicitly, then fill via kernel.
        expert_offsets[0] = 0

        BLOCK2 = 256  # one program per 256-expert chunk
        grid2 = (triton.cdiv(num_experts, BLOCK2),)
        prefix_offsets_kernel[grid2](counts, expert_offsets, num_experts, BLOCK=BLOCK2, num_warps=1)

        # 3) Stable sort of flattened indices (PyTorch, values-only, not tied to num_experts)
        sorted_token_indices = flat_i32.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
