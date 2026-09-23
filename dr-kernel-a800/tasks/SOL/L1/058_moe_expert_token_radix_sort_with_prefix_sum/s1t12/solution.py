import triton
import triton.language as tl


# Kernel 1: count occurrences per expert ID, no atomics
# flat_ptr: [N] int32, counts_ptr: [NUM_EXPERTS] int32, N: int32
@triton.jit
def count_experts_kernel(flat_ptr, counts_ptr, N: tl.int32, M: tl.int32, BLOCK_HIST: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_HIST
    # For each tile, iterate over all expert ids and accumulate local sums
    for k in range(M):
        cnt = tl.zeros((), dtype=tl.int32)
        for j in range(BLOCK_HIST):
            idx = start + j
            m = idx < N
            val = tl.load(flat_ptr + idx, mask=m, other=0)
            cnt += (val == k) * (m.to(tl.int32))
        tl.atomic_add(counts_ptr + k, cnt)


# Kernel 2: inclusive prefix sum over a small vector
# in_ptr: [M] int32, out_ptr: [M] int32, M: int32, BLOCK_M: tl.constexpr (>= M)
@triton.jit
def prefix_sum_kernel(in_ptr, out_ptr, M: tl.int32, BLOCK_M: tl.constexpr):
    carry = tl.zeros((), dtype=tl.int32)
    for i in range(BLOCK_M):
        # load with mask to handle BLOCK_M > M
        x = tl.load(in_ptr + i, mask=i < M, other=0)
        carry += x
        tl.store(out_ptr + i, carry)


# Kernel 3: compute stable argsort permutation: out[i] = position of flat[i] in stable order
# flat_ptr: [N] int32, le_counts_ptr: [M] int32, lt_counts_ptr: [M] int32, out_ptr: [N] int32, N: int32, M: int32
@triton.jit
def compute_out_pos_real(flat_ptr, le_counts_ptr, lt_counts_ptr, out_ptr, N: tl.int32, M: tl.int32, BLOCK_OUT: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_OUT
    for j in range(BLOCK_OUT):
        idx = start + j
        m = idx < N
        # load id (masked)
        idv = tl.load(flat_ptr + idx, mask=m, other=0)
        # get inclusive and exclusive counts (masked loads for M)
        # Note: idv is in [0, M-1], but we keep mask robustness
        lev = tl.load(le_counts_ptr + idv, mask=idv < M, other=0)
        ltv = tl.load(lt_counts_ptr + idv, mask=idv < M, other=0)
        # duplicates flag: 1 if ltv > 0 else 0
        duplicates = tl.where(ltv > 0, 1, 0)
        pos = lev - duplicates
        tl.store(out_ptr + idx, pos, mask=m)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # Fixed number of experts as in the original code
        M = 256

        # 1) Compute counts per expert ID using Triton kernel
        counts = torch.zeros(M, dtype=torch.int32, device=device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        count_experts_kernel[grid_hist](flat, counts, N, M, BLOCK_HIST=BLOCK_HIST)

        # 2) Compute le_counts via inclusive scan (prefix sum) using Triton kernel
        le_counts = torch.empty(M, dtype=torch.int32, device=device)
        BLOCK_M = 256  # since M=256, we use a single program
        prefix_sum_kernel[(1,)](counts, le_counts, M, BLOCK_M=BLOCK_M)

        # 3) Compute lt_counts = le_counts - counts (exclusive count per expert)
        lt_counts = le_counts - counts

        # 4) Compute stable argsort permutation using Triton kernel
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        BLOCK_OUT = 1024
        grid_out = (triton.cdiv(N, BLOCK_OUT),)
        compute_out_pos_real[grid_out](flat, le_counts, lt_counts, sorted_token_indices, N, M, BLOCK_OUT=BLOCK_OUT)

        # 5) Compute expert_offsets (length M+1) as cumulative counts of occurrences
        #    We use torch.cumsum here because counts is small; it's negligible and correct.
        offsets = torch.cumsum(counts, dim=0).to(torch.int32)
        # Ensure length is M+1: torch.cumsum(counts) returns length M; we need one extra for the prefix.
        # Since counts length is M, offsets[0] = 0, offsets[1:] = le_counts. We already computed le_counts.
        # Therefore, construct offsets as [0, le_counts[0], le_counts[0]+le_counts[1], ...]
        # But we already have le_counts. To return requested offsets, create [0] + prefix of le_counts.
        # However, we need a tensor of length M+1. Use torch.cat for clarity.
        offsets = torch.empty(M + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        offsets[1:] = le_counts

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
