import triton
import triton.language as tl


@triton.jit
def histogram_atomic_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Histogram of values in flat_ptr (int32), one pass with atomic adds.
    flat_ptr: *int32, shape [N]
    counts_ptr: *int32, shape [num_experts]
    N: int, total number of elements
    Each program processes BLOCK elements; masked loads prevent OOB.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N
    # Iterate scalarly with mask; Triton will vectorize across offsets.
    for i in range(BLOCK):
        idx = start + i
        m = mask[i]
        if m:
            id_val = tl.load(flat_ptr + idx)
            tl.atomic_add(counts_ptr + id_val, 1)


@triton.jit
def compute_le_counts(counts_ptr, le_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr into le_ptr for k in [0, num_experts-1].
    counts_ptr: *int32, shape [num_experts]
    le_ptr: *int32, shape [num_experts]
    Single-program reduction loop.
    """
    running = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_experts):
        v = tl.load(counts_ptr + k)
        running += v
        tl.store(le_ptr + k, running)


@triton.jit
def compute_lt_counts(le_ptr, counts_ptr, lt_ptr, num_experts: tl.int32):
    """
    Compute lt_counts[k] = le_counts[k] - counts[k] for k in [0, num_experts-1].
    le_ptr: *int32, shape [num_experts]
    counts_ptr: *int32, shape [num_experts]
    lt_ptr: *int32, shape [num_experts]
    Single-program loop.
    """
    for k in range(0, num_experts):
        le_k = tl.load(le_ptr + k)
        cnt_k = tl.load(counts_ptr + k)
        tl.store(lt_ptr + k, le_k - cnt_k)


@triton.jit
def compute_out_pos(flat_ptr, out_ptr, le_ptr, lt_ptr, N, num_experts: tl.int32):
    """
    Compute stable argsort permutation of flat_ptr (length N), writing to out_ptr.
    Uses le_counts (inclusive) and lt_counts (exclusive <) per value.
    For each element i:
      id = flat[i]
      pos = le[id] - (1 if duplicates else 0)
      out[pos] = i
    Launch with grid = (triton.cdiv(N, 1),) and mask to avoid OOB.
    """
    pid = tl.program_id(axis=0)
    offsets = pid  # one element per program
    mask = offsets < N
    if mask:
        id_val = tl.load(flat_ptr + offsets)
        le_k = tl.load(le_ptr + id_val)
        lt_k = tl.load(lt_ptr + id_val)
        # duplicate flag: if counts[id] > 1, subtract 1 for stable tie-breaking
        cnt_k = tl.load(le_ptr + id_val) - lt_k  # le_k - cnt_k = lt_k, so cnt_k = le_k - lt_k
        duplicate = (cnt_k > 1)
        pos = le_k - (1 if duplicate else 0)
        tl.store(out_ptr + pos, offsets)


@triton.jit
def prefix_scan_expert_offs(counts_ptr, offsets_ptr, num_experts: tl.int32):
    """
    Compute inclusive prefix sum of counts_ptr and write to offsets_ptr[1:].
    offsets_ptr[0] must be set to 0 on host.
    Single-program loop.
    """
    running = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_experts):
        v = tl.load(counts_ptr + k)
        running += v
        tl.store(offsets_ptr + k + 1, running)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts fixed at 256, matching the original code
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure we operate on CUDA device; get N and flat
        assert topk_idx.is_cuda, "Input tensor must be on CUDA device for Triton."
        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Histogram of expert IDs via Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_atomic_kernel[grid_hist](flat, counts, N, BLOCK_HIST)

        # 2) Compute le_counts (inclusive) and lt_counts (exclusive <) via Triton
        le_counts = torch.empty(self.num_experts, dtype=torch.int32, device=flat.device)
        compute_le_counts[(1,)](counts, le_counts, self.num_experts)

        lt_counts = torch.empty(self.num_experts, dtype=torch.int32, device=flat.device)
        compute_lt_counts[(1,)](le_counts, counts, lt_counts, self.num_experts)

        # 3) Compute stable argsort permutation via Triton (launch required kernel)
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=flat.device)
        # One program per element; masks guard OOB. Using grid=(N,) is fine; Triton will handle.
        compute_out_pos[(N,)](flat, sorted_token_indices, le_counts, lt_counts, N, self.num_experts)

        # 4) Compute expert_offsets via Triton prefix scan
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0
        prefix_scan_expert_offs[(1,)](counts, expert_offsets, self.num_experts)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
