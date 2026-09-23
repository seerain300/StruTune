import torch
import triton
import triton.language as tl


@triton.jit
def count_histogram_kernel(flat_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel: compute histogram of flat values (int32) into counts_ptr (int32).
    counts_ptr length = num_experts. flat_ptr is 1D int32 of length N.
    Each program processes BLOCK elements and atomically adds 1 for each occurrence.
    """
    pid = tl.program_id(axis=0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32 values in [0, num_experts-1]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Triton kernel: compute inclusive prefix sum of counts_ptr (int32) into offsets_ptr (int64),
    writing offsets[1..num_experts]. offsets_ptr[0] should be set by the caller to 0.
    Single program performs sequential loop.
    """
    acc = tl.zeros((), dtype=tl.int64)
    for i in range(0, num_experts):
        val = tl.load(counts_ptr + i)
        acc += val
        tl.store(offsets_ptr + (i + 1), acc)


@triton.jit
def stable_bitonic_sort_kernel(flat_ptr, out_idx_ptr, N, LOGN: tl.constexpr):
    """
    Triton kernel: stable bitonic sort on flat_ptr (int32). out_idx_ptr holds original positions 0..N-1 as int64.
    Axis 0: N programs (one per element).
    Axis 1: LOGN programs (one per stage j in bitonic network).
    For each stage j:
      partner = i ^ (1 << j)
      Only process pairs where i < partner to avoid races.
      Ascending if ((i & (1 << (j+1))) == 0), else descending.
      Stability tie-break: if equal, lower original index (i < partner) comes first.
    """
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    # Compute partner for this stage j
    partner = i ^ (1 << j)
    do_pair = i < partner
    in_range = (i < N) & (partner < N) & do_pair

    # Load current values
    a = tl.load(flat_ptr + i)
    b = tl.load(flat_ptr + partner)
    idx_a = tl.load(out_idx_ptr + i)
    idx_b = tl.load(out_idx_ptr + partner)

    # Direction: if bit (j+1) of i is 0 -> ascending, else descending
    asc = (tl.bitcast(i, tl.int32) & (1 << (j + 1))) == 0

    # Compare
    cond = (a > b) if asc else (a < b)
    # Stability: if equal, lower index comes first
    cond = cond | ((a == b) & (idx_a > idx_b))

    new_idx_a = idx_b if cond else idx_a
    new_idx_b = idx_a if cond else idx_b

    # Only write for the lower index in the pair
    if do_pair:
        tl.store(out_idx_ptr + i, new_idx_a)
        # partner program will write out_idx_ptr[partner] = new_idx_b


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32, values in [0, num_experts-1]
        Returns:
          sorted_token_indices: (N,), int64
          expert_offsets: (num_experts + 1,), int32
        """
        # Ensure contiguous and flatten
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        topk_idx = topk_idx.contiguous()
        flat = topk_idx.reshape(-1)  # 1D int32
        N = flat.numel()

        # 1) Compute histogram in Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)
        BLOCK = 1024
        grid_counts = (triton.cdiv(N, BLOCK),)
        count_histogram_kernel[grid_counts](flat, counts, N, num_experts=self.num_experts)

        # 2) Compute expert_offsets in Triton
        # Allocate int64 offsets buffer; set offset[0] = 0 on host; kernel writes offsets[1..]
        offsets_long = torch.empty(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        offsets_long[0] = 0
        # Run Triton prefix sum kernel
        prefix_sum_kernel[(1,)](counts, offsets_long, num_experts=self.num_experts)
        expert_offsets = offsets_long[1:].to(torch.int32)

        # 3) Stable sort in Triton
        # Initialize out_idx with original positions as int64
        out_idx = torch.empty(N, dtype=torch.int64, device=flat.device)
        # LOGN = ceil(log2(N)); pass as tl.constexpr
        LOGN = int(torch.ceil(torch.log2(torch.tensor(N, device=flat.device)))).item()

        grid_sort = (N, LOGN)
        stable_bitonic_sort_kernel[grid_sort](flat, out_idx, N, LOGN)

        sorted_token_indices = out_idx

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
