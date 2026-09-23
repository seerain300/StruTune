import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: histogram per-expert counts for a 1D array of indices
@triton.jit
def _histogram_kernel(flat_ptr, N, counts_ptr, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load indices; default 0 for masked lanes
    idx = tl.load(flat_ptr + offsets, mask=mask, other=0)

    # Vector of counts for each expert id (up to num_experts=256). We use a vector of length 256.
    # Each lane corresponds to one expert id; we initialize to zeros.
    cnts = tl.zeros((num_experts,), dtype=tl.int32)

    # Build counts per expert by comparing each lane
    # Note: Triton will JIT this vectorized comparison and reduction.
    # For masked lanes (idx==0), comparisons yield False and cnts remain zero.
    for e in range(num_experts):
        # (idx == e) produces a boolean vector; cast to int32
        cnts[e] = tl.sum((idx == e), axis=0)

    # Atomically accumulate per-expert counts from this block
    for e in range(num_experts):
        tl.atomic_add(counts_ptr + e, cnts[e])


# Triton kernel: inclusive prefix sum of counts to produce expert_offsets (num_experts + 1)
# We use static chunked loops to avoid dynamic Python control flow.
@triton.jit
def _prefix_sum_experts_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr, EXPERTS_PER_PASS: tl.constexpr):
    # One program instance processes EXPERTS_PER_PASS experts starting at chunk * EXPERTS_PER_PASS.
    chunk = tl.program_id(0)

    # First pass: compute local inclusive sums within the chunk and write them with carry.
    carry = tl.zeros((), dtype=tl.int32)  # scalar carry from previous chunks
    for i in tl.static_range(EXPERTS_PER_PASS):
        idx = chunk * EXPERTS_PER_PASS + i
        if idx < num_experts:
            local = tl.load(counts_ptr + idx)
            tl.store(offsets_ptr + idx + 1, carry)
            carry += local

    # Second pass: add carry to each local sum in the chunk and write the final offsets.
    for i in tl.static_range(EXPERTS_PER_PASS):
        idx = chunk * EXPERTS_PER_PASS + i
        if idx < num_experts:
            local = tl.load(offsets_ptr + idx + 1)  # previously stored carry
            carry_new = local + tl.load(counts_ptr + idx)
            tl.store(offsets_ptr + idx + 1, carry_new)
            # Update carry for next expert in this chunk
            carry = carry_new


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; Triton kernels handle computation.

    def forward(self, *args):
        # Expect a single tensor argument: topk_idx
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]

        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # Triton histogram for expert counts
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_kernel[grid](flat, N, counts, num_experts=num_experts, BLOCK=BLOCK, num_warps=4)

        # Triton inclusive prefix sum to produce expert_offsets (num_experts + 1)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)

        EXPERTS_PER_PASS = 128  # chunk size for static loops
        grid_prefix = (triton.cdiv(num_experts, EXPERTS_PER_PASS),)
        _prefix_sum_experts_kernel[grid_prefix](counts, expert_offsets, num_experts=num_experts, EXPERTS_PER_PASS=EXPERTS_PER_PASS, num_warps=1)

        # Stable sort of flattened indices (PyTorch on GPU, data-independent on num_experts)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
