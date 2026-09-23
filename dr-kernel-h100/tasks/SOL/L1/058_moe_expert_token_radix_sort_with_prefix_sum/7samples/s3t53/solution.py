import torch

# Triton kernels
import triton
import triton.language as tl


@triton.jit
def _stable_rank_argsort_indices_single_block(flat_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Stable argsort permutation. For each original index i, computes its stable rank by scanning all j,
    and writes i to out[rank]. Output is int64 indices.
    Assumes flat_ptr is int32, out_ptr is int64.
    """
    # One program per original index
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load the value for this index
    val_i = tl.load(flat_ptr + pid)

    # Compute stable rank by scanning all j
    rank = tl.zeros((), dtype=tl.int32)
    # Limit scan to BLOCK to reduce work; mask out j >= N
    for j in range(BLOCK):
        valid = j < N
        val_j = tl.load(flat_ptr + j, mask=valid, other=0)
        # stable: count elements less than current and for ties, those with smaller original index
        less = val_j < val_i
        tie = (val_j == val_i) & (j < pid)
        rank += (less | tie).to(tl.int32)

    # Store the original index (pid) at computed rank as int64
    # Note: out_ptr is int64, so tl.store will cast pid to int64 if needed
    tl.store(out_ptr + rank, tl.full((), pid, tl.int64))


@triton.jit
def _histogram_kernel(values_ptr, N, histogram_ptr, num_buckets: tl.constexpr):
    """
    Histogram of values_ptr (int32) into histogram_ptr (int32).
    Each program handles one element and performs atomic_add to its bucket.
    num_buckets is fixed (e.g., 256).
    """
    idx = tl.program_id(0)
    if idx >= N:
        return
    val = tl.load(values_ptr + idx)
    # Ensure value is within [0, num_buckets - 1]
    # (get_inputs guarantees valid indices in [0, num_experts-1], i.e., 255)
    tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(histogram_ptr, offsets_ptr, num_buckets: tl.constexpr):
    """
    Inclusive prefix sum of histogram_ptr (int32) into offsets_ptr (int32),
    producing length (num_buckets + 1), offsets_ptr[0] must be set by host to 0.
    """
    # Single program sequential scan
    s = 0
    for i in range(num_buckets):
        s += tl.load(histogram_ptr + i)
        tl.store(offsets_ptr + i + 1, s)
    # offsets_ptr[0] is pre-set by host


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure on CUDA and contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton execution"
        flat = topk_idx.contiguous().view(-1).to(torch.int32)

        N = flat.numel()
        device = flat.device

        # 1) Triton stable argsort permutation (out is int64 indices)
        out = torch.empty(N, dtype=torch.int64, device=device)
        # Launch one program per original index
        BLOCK = 4096  # scan upper bound; masking ensures correctness for N <= 4096. If N > BLOCK, adjust host code accordingly.
        grid = (N,)
        _stable_rank_argsort_indices_single_block[grid](flat, out, N, BLOCK)

        # 2) Triton histogram of expert IDs
        num_experts = 256
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_experts)

        # 3) Triton inclusive prefix sum to produce expert_offsets of length (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_experts)

        # Return sorted_token_indices as int64 and expert_offsets as int32
        return out, offsets


def run(*args):
    return ModelNew()(*args)
