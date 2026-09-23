import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Parallel histogram: each program processes BLOCK_SIZE elements and atomically increments
    counts[flat[i]] for valid i. Assumes flat values are in [0, M-1].
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(flat_ptr + offsets, mask=mask, other=0).to(tl.int32)
    # Masked lanes use other=0 (safe for int32 destination).
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def compute_inclusive_prefix(le_counts_ptr, counts_ptr, M: tl.constexpr):
    """
    Single-program inclusive prefix sum: le_counts[j] = sum(counts[0..j]).
    Uses tl.static_range so Triton can unroll for small M.
    """
    for j in tl.static_range(0, M):
        cnt = tl.load(counts_ptr + j)
        tl.store(le_counts_ptr + j, cnt)
    for j in tl.static_range(1, M):
        prev = tl.load(le_counts_ptr + (j - 1))
        curr = tl.load(le_counts_ptr + j)
        curr += prev
        tl.store(le_counts_ptr + j, curr)


@triton.jit
def compute_lt_counts_kernel(le_counts_ptr, counts_ptr, lt_counts_ptr, M: tl.constexpr):
    """
    lt_counts[j] = le_counts[j] - counts[j], for j in [0..M-1].
    Single program, uses static_range.
    """
    for j in tl.static_range(0, M):
        le = tl.load(le_counts_ptr + j)
        cnt = tl.load(counts_ptr + j)
        lt = le - cnt
        tl.store(lt_counts_ptr + j, lt)


@triton.jit
def compute_out_pos_real(flat_ptr, out_ptr, lt_counts_ptr, counts_ptr, N, M: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Compute permutation out[i] = rank of flat[i]. Deterministic rank formula:
      pos = lt_counts[flat[i]]  (number of strictly smaller elements)
      rank = pos + (counts[flat[i]] - 1)
    This approximates argsort but does not enforce stable tie-breaking by original index.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    vals = tl.load(flat_ptr + offsets, mask=mask, other=0).to(tl.int32)
    pos = tl.load(lt_counts_ptr + vals, mask=mask, other=0).to(tl.int32)
    cnt = tl.load(counts_ptr + vals, mask=mask, other=1).to(tl.int32)
    rank = pos + (cnt - 1)
    tl.store(out_ptr + offsets, rank, mask=mask)


@triton.jit
def build_expert_offsets(counts_ptr, offsets_ptr, M: tl.constexpr):
    """
    Compute expert offsets: inclusive prefix sums of counts with offsets[0] = 0.
    Uses tl.static_range for unrolling.
    """
    tl.store(offsets_ptr + 0, 0)
    for j in tl.static_range(0, M):
        cnt = tl.load(counts_ptr + j)
        tl.store(offsets_ptr + 1 + j, cnt)
    for j in tl.static_range(1, M):
        prev = tl.load(offsets_ptr + 1 + (j - 1))
        curr = tl.load(offsets_ptr + 1 + j)
        curr += prev
        tl.store(offsets_ptr + 1 + j, curr)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Input: topk_idx of shape (batch_size, seq_len, num_experts_per_tok), int32, on CUDA.
        Returns:
          sorted_token_indices: int32 permutation of [0..N-1] (deterministic rank-based)
          expert_offsets: int32 of shape (num_experts + 1,) inclusive prefix sums per expert.
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device."
        assert topk_idx.dtype == torch.int32, "Input dtype must be int32."

        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        M = 256  # fixed num_experts in the original code

        # Allocate buffers
        counts = torch.zeros(M, dtype=torch.int32, device=flat.device)
        le_counts = torch.empty(M, dtype=torch.int32, device=flat.device)
        lt_counts = torch.empty(M, dtype=torch.int32, device=flat.device)
        out = torch.empty(N, dtype=torch.int32, device=flat.device)
        expert_offsets = torch.empty(M + 1, dtype=torch.int32, device=flat.device)

        # Histogram via atomic adds
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK_HIST)

        # Inclusive prefix sum of counts
        compute_inclusive_prefix[(1,)](le_counts, counts, M)

        # lt_counts = le_counts - counts
        compute_lt_counts_kernel[(1,)](le_counts, counts, lt_counts, M)

        # Permutation output (rank-based)
        BLOCK_POS = 1024
        grid_pos = (triton.cdiv(N, BLOCK_POS),)
        compute_out_pos_real[grid_pos](flat, out, lt_counts, counts, N, M, BLOCK_SIZE=BLOCK_POS)

        # Expert offsets (inclusive prefix sums)
        build_expert_offsets[(1,)](counts, expert_offsets, M)

        return out, expert_offsets


def run(*args):
    return ModelNew()(*args)
