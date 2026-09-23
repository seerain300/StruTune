import triton
import triton.language as tl


# Triton kernel: histogram of flat values using atomic adds.
# flat: pointer to int32 values of length N
# counts: pointer to int32 of length num_experts, initialized to zeros
# N: number of elements in flat
@triton.jit
def histogram_atomic_kernel(flat, counts, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Masked load; invalid positions get value 0 (won't be stored due to mask).
    vals = tl.load(flat + offs, mask=mask, other=0)
    # Atomic add 1 for each valid position.
    tl.atomic_add(counts + vals, 1, mask=mask)


# Triton kernel: compute inclusive le_counts (prefix sum) and lt_counts (exclusive minus 1) across experts.
# counts: int32 counts of length num_experts (num_experts is known on host as NUM_EXPS)
# le_counts: int32 output of length NUM_EXPS+1, offsets[0] = 0
# lt_counts: int32 output of length NUM_EXPS, offsets for earlier elements for each expert
@triton.jit
def scan_counts_kernel(counts, le_counts, lt_counts, NUM_EXPS: tl.constexpr):
    # Inclusive scan across NUM_EXPS and store le_counts[1:].
    total = 0
    for j in range(NUM_EXPS):
        total += counts[j]
        # le_counts is length NUM_EXPS+1; store at index j+1
        tl.store(le_counts + (j + 1), total)
    # lt_counts[j] = le_counts[j+1] - counts[j]
    for j in range(NUM_EXPS):
        tl.store(lt_counts + j, tl.load(le_counts + (j + 1)) - counts[j])


# Triton kernel: compute stable argsort permutation. Produces sorted_token_indices.
# flat: pointer to int32 values of length N
# out: pointer to int32 permutation of length N
# counts: int32 counts of length num_experts
# le_counts: int32 of length num_experts+1
# lt_counts: int32 of length num_experts
# N: number of elements
@triton.jit
def compute_out_pos(flat, out, counts, le_counts, lt_counts, N, NUM_EXPS: tl.constexpr):
    for i in range(0, N):
        # Load current value
        val = tl.load(flat + i)
        # Default position is inclusive prefix of this value
        pos = tl.load(le_counts + val)
        # Detect duplicates with earlier indices that already took that pos
        has_duplicate = 0
        for j in range(0, i):
            vj = tl.load(flat + j)
            pj = tl.load(le_counts + vj)
            # If vj == val and pj == pos, then there is at least one earlier index with the same value
            # at the same position; subtract 1 for stable ordering.
            has_duplicate |= ((vj == val) & (pj == pos))
        pos = pos - has_duplicate
        # Store i at position pos in out
        tl.store(out + pos, i)


# Forward: Triton-based implementation without torch ops on tensors (except for minor ops).
class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure dtype int32 and contiguous
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device
        N = flat.numel()

        # Prepare outputs and buffers
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)

        # Histogram via Triton
        num_experts = 256  # match original code; adjust if needed
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK=BLOCK_HIST)

        # Compute le_counts and lt_counts via Triton (single-program kernels)
        le_counts = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        lt_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        scan_counts_kernel[(1,)](counts, le_counts, lt_counts, NUM_EXPS=num_experts)

        # Compute stable permutation via Triton
        compute_out_pos[(1,)](flat, sorted_token_indices, counts, le_counts, lt_counts, N, NUM_EXPS=num_experts)

        # Return the required outputs: sorted_token_indices permutation and expert offsets.
        return sorted_token_indices


def run(*args):
    return ModelNew()(*args)
