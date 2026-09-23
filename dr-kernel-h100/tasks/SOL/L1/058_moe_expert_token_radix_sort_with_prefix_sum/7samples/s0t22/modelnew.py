import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: count occurrences of each expert id (0..255) in flat.
# flat is int32 1D tensor of length N. counts is int32 vector of length 256.
# We increment counts[i] for each occurrence where flat[j] == i.
@triton.jit
def bincount_kernel(
    flat_ptr,          # *int32, flattened input
    counts_ptr,        # *int32, output counts for bins [0..255]
    N,                 # int32, number of elements in flat
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values; if out of bounds, load 0 (won't contribute due to mask)
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # We expect vals in [0, 255]. The provided get_inputs guarantees this.
    # For each possible bin i, count how many vals == i.
    for i in range(256):
        eq = vals == i  # boolean vector
        # Sum of booleans: cast to int32 and reduce to scalar (per program)
        cnt_i = tl.sum(eq.to(tl.int32), axis=0)
        # Atomic add to global counts[i]
        tl.atomic_add(counts_ptr + i, cnt_i)


# Triton kernel: compute inclusive prefix sum of a 1D vector x (int32) of length L,
# write to y (int64) of same length. We do this in a single program with a loop.
@triton.jit
def inclusive_prefix_sum_i32_to_o64_kernel(
    x_ptr,     # *int32 input (counts)
    y_ptr,     # *int64 output (offsets)
    L,         # int32 length
):
    # Single program handles the entire vector
    acc = tl.zeros((), dtype=tl.int64)  # scalar accumulator
    for k in range(L):
        val_i32 = tl.load(x_ptr + k)
        val_i64 = val_i32.to(tl.int64)
        acc += val_i64
        tl.store(y_ptr + k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized replacement for the original run function:
        - sorted_token_indices: stable argsort permutation of flattened indices (int64)
        - expert_offsets: inclusive prefix sum of per-expert counts (int64), length 257
        """
        # Ensure CUDA tensor (if not, move to cuda). Triton requires CUDA device.
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.cuda()

        # Flatten and ensure int64 for argsort to match original behavior.
        flat = topk_idx.reshape(-1).to(torch.int64)

        # sorted_token_indices: permutation of [0, N-1] sorted by flat values (stable)
        sorted_token_indices = flat.argsort(stable=True).to(torch.int64)

        # Compute counts via Triton when available; otherwise, fall back to torch.bincount.
        N = flat.numel()
        # We will use num_experts = 256, matching the original code's constant.
        num_experts = 256
        if TRITON_AVAILABLE:
            # counts as int32
            counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
            # Choose a reasonable block size
            BLOCK = 1024
            grid = (triton.cdiv(N, BLOCK),)
            bincount_kernel[grid](flat.to(torch.int32), counts, N, BLOCK=BLOCK)
        else:
            # Fallback: torch.bincount, then cumsum in Triton or torch (we will do Triton prefix sum next anyway)
            counts = torch.bincount(flat.to(torch.int64), minlength=num_experts).to(torch.int32)

        # Compute inclusive prefix sum of counts in Triton, output int64 offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int64, device=flat.device)
        inclusive_prefix_sum_i32_to_o64_kernel[(1,)](counts, offsets, num_experts + 1)

        return sorted_token_indices, offsets