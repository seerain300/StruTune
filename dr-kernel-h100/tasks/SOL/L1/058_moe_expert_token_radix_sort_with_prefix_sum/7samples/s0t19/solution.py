import torch

# Try to import Triton; use it if available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel 1: count occurrences of each expert id in [0, 255]
# x_ptr: flattened int32 topk_idx, length N
# counts_ptr: output int32 vector of length 256
if TRITON_AVAILABLE:
    @triton.jit
    def bincount_kernel(x_ptr, counts_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(x_ptr + offs, mask=mask, other=0)  # int32
        # For each id in [0, 255], sum how many times it appears among vals under mask
        for i in range(256):
            eq_mask = (vals == i) & mask
            sum_i = tl.sum(eq_mask.to(tl.int32), axis=0)
            tl.atomic_add(counts_ptr + i, sum_i)

# Triton kernel 2: inclusive prefix sum over a 256-length counts vector, producing int64 offsets of length 257
# x_ptr: int32 counts vector, length 256
# y_ptr: int64 output vector, length 257 (we write inclusive sums at indices 1..256; 0 left unused)
if TRITON_AVAILABLE:
    @triton.jit
    def inclusive_prefix_sum_kernel(counts_ptr, y_ptr, L: tl.constexpr):
        # L should be 257: 256 counts + 1 for final inclusive sum
        # We compute inclusive sums for indices 1..256 into y_ptr[1..256]
        cur_sum = tl.zeros((), dtype=tl.int64)
        # Unroll the loop manually (compile-time constexpr). Triton will generate code for these iterations.
        # For robustness, we only compute up to index 256 (since L=257). The original code uses num_experts=256.
        for j in range(256):
            val_j = tl.load(counts_ptr + j).to(tl.int64)
            cur_sum = cur_sum + val_j
            # Write inclusive sum at position j+1 in y_ptr
            tl.store(y_ptr + (j + 1), cur_sum)

# Entry point class: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA for Triton
        device = torch.device("cuda")
        topk_idx = topk_idx.to(device)

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Triton bincount into int32 counts vector of length 256
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        if TRITON_AVAILABLE:
            BLOCK = 1024
            grid = (triton.cdiv(N, BLOCK),)
            bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)
        else:
            # Fallback: torch.bincount for correctness if Triton not available
            counts = torch.bincount(flat, minlength=256)

        # 2) Inclusive prefix sum in Triton, output int64 of length 257
        expert_offsets = torch.empty(257, dtype=torch.int64, device=device)
        if TRITON_AVAILABLE:
            inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, L=257)
        else:
            # Fallback: torch.cumsum for correctness
            expert_offsets = counts.cumsum(0).to(torch.int64)

        # 3) sorted_token_indices: stable argsort of flattened indices (matches original behavior)
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
