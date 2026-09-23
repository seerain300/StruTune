import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(
    x_ptr,                  # *int32, flattened input
    counts_ptr,             # *int32, length num_experts
    N,                      # int32 total number of elements (runtime)
    num_experts: tl.constexpr,  # compile-time constant (256)
    BLOCK_HIST: tl.constexpr,   # block size for vectorized load
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_HIST + tl.arange(0, BLOCK_HIST)
    mask = offsets < N
    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    # atomic add 1 for each valid element
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def scan_exclusive_counts_kernel(
    counts_ptr,             # *int32, length num_experts
    le_counts_ptr,          # *int32, length num_experts
    lt_counts_ptr,          # *int32, length num_experts
    num_experts: tl.constexpr,
):
    # Single-program exclusive prefix sum over counts
    prefix = 0
    for i in range(0, num_experts):
        count_i = tl.load(counts_ptr + i)
        lt_counts_ptr[i] = prefix
        le_counts_ptr[i] = prefix + count_i
        prefix = prefix + count_i


@triton.jit
def compute_out_pos(
    x_ptr,                  # *int32, flattened input
    out_ptr,                # *int32, length N (sorted_token_indices)
    N,                      # int32 total number of elements (runtime)
    num_experts: tl.constexpr,    # 256
    le_counts_ptr,          # *int32, length num_experts
):
    # Compute stable permutation. For each output index i, we find k = x[i],
    # then compute pos = le_counts[k] - (1 if any earlier j has same value else 0).
    # This enforces stable tie-breaking (earlier index comes first for duplicates).
    for i in range(0, N):
        val = tl.load(x_ptr + i)              # value at position i
        le_val = tl.load(le_counts_ptr + val) # inclusive count up to val
        # Initialize has_sub = 0
        has_sub = 0
        # Check for any earlier j < i with same value and out[j] == le_val - 1
        # Note: Triton doesn't support dynamic break; we loop up to N-1 and accumulate has_sub.
        for j in range(0, N):
            if j < i:
                val_j = tl.load(x_ptr + j)
                pos_j = tl.load(out_ptr + j)  # only used to detect stable occurrence
                if val_j == val and pos_j == (le_val - 1):
                    has_sub = 1
        pos = le_val - has_sub
        tl.store(out_ptr + i, pos)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx: int32 tensor shaped (B, S, L)
        device = topk_idx.device
        flat = topk_idx.reshape(-1).contiguous()  # shape [N], int32
        N = flat.numel()
        num_experts = 256  # fixed as in original code

        # 1) Histogram of values via Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, num_experts, BLOCK_HIST)

        # 2) Exclusive prefix sum to produce le_counts
        le_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        # lt_counts is not needed here; we only need le_counts for final position
        scan_exclusive_counts_kernel[(1,)](counts, le_counts, torch.empty_like(le_counts), num_experts)

        # 3) Compute stable argsort permutation via Triton
        out = torch.empty(N, dtype=torch.int32, device=device)
        compute_out_pos[(1,)](flat, out, N, num_experts, le_counts)

        # Return sorted_token_indices (permutation indices)
        return out


def run(*args):
    return ModelNew()(*args)
