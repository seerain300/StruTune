import triton
import triton.language as tl


# Triton kernel: build counts per expert id using atomic adds (masked).
@triton.jit
def histogram_atomic_kernel(
    flat_ptr,       # *int32, flattened input
    counts_ptr,     # *int32, output counts vector of length num_experts
    N,              # int32, total number of elements in flat
    BLOCK_HIST: tl.constexpr  # tile size
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_HIST + tl.arange(0, BLOCK_HIST)
    mask = offs < N
    ids = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Atomic add 1 for each valid id into counts_ptr[ids]
    tl.atomic_add(counts_ptr + ids, 1, mask=mask)


# Triton kernel: set duplicate_flag[k] = 1 if counts[k] > 1, else 0.
@triton.jit
def set_duplicate_flags(counts_ptr, duplicate_ptr, num_experts: tl.constexpr):
    for k in range(0, num_experts):
        c = tl.load(counts_ptr + k)
        is_dup = c > 1
        tl.store(duplicate_ptr + k, is_dup.to(tl.int32))


# Triton kernel: compute stable argsort permutation. One program per element.
# For element ii (0..N-1), id = flat[ii], pos = le_counts[id] - (1 if duplicate[id] and ii > lt_counts[id]).
@triton.jit
def compute_out_pos_real(
    flat_ptr,        # *int32, flattened input
    out_pos_ptr,     # *int32, output permutation of length N
    N,               # int32
    num_experts: tl.constexpr,  # number of unique experts
    le_counts_ptr,   # *int32, length num_experts, inclusive prefix sums
    lt_counts_ptr,   # *int32, length num_experts, inclusive prefix minus counts
    duplicate_flag_ptr,  # *int32, length num_experts, 0/1
    BLOCK_N: tl.constexpr  # we set 1 (one element per program)
):
    ii = tl.program_id(axis=0)
    if ii >= N:
        return
    id_val = tl.load(flat_ptr + ii)
    # Bounds safety (shouldn't happen if ids in [0, num_experts-1], but guard anyway)
    if id_val >= num_experts:
        # default to 0 for le_counts (won't be used)
        le_k = 0
        lt_k = 0
        dup_flag = 0
    else:
        le_k = tl.load(le_counts_ptr + id_val)
        lt_k = tl.load(lt_counts_ptr + id_val)
        dup_flag = tl.load(duplicate_flag_ptr + id_val)
    # Stable position: subtract 1 if duplicate and this element's index is greater than the count of strictly smaller ids
    need_sub = (dup_flag != 0) & (ii > lt_k)
    pos = le_k - need_sub.to(tl.int32)
    tl.store(out_pos_ptr + ii, pos)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure contiguous int32 on CUDA device
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device
        assert device.type == 'cuda', "ModelNew expects CUDA tensors."

        # Determine num_experts from input: original code uses num_experts = topk_idx.max() + 1
        # Cast to int to use as Triton constexpr
        max_id = int(flat.max().item())
        num_experts = max_id + 1

        # 1) Histogram per expert id using Triton
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK_HIST)

        # 2) Precompute le_counts (inclusive prefix) and lt_counts (inclusive prefix - counts) using PyTorch ops on device
        # Compute inclusive prefix sums: le_counts[k] = sum_{j<=k} counts[j]
        le_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        running = 0
        for k in range(num_experts):
            running += int(counts[k].item())
            le_counts[k] = running
        lt_counts = le_counts - counts  # per-id number of strictly smaller ids

        # 3) Duplicate flags per id (used for stable tie-breaking)
        duplicate_flag = torch.zeros(num_experts, dtype=torch.int32, device=device)
        set_duplicate_flags[(1,)](counts, duplicate_flag, num_experts)

        # 4) Compute stable argsort permutation using Triton: one program per element
        out_pos = torch.empty(N, dtype=torch.int32, device=device)
        grid_out = (N,)
        compute_out_pos_real[grid_out](flat, out_pos, N, num_experts, le_counts, lt_counts, duplicate_flag, 1)

        # 5) Compute expert offsets (inclusive prefix sums of counts), matching original behavior
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[1:] = counts.cumsum(0)

        # Return outputs: sorted_token_indices (out_pos) and expert_offsets
        return out_pos, expert_offsets


def run(*args):
    return ModelNew()(*args)
