import torch
import triton

@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ids = tl.load(x_ptr + offs, mask=mask, other=0)
    # Increment counts for each id in [0, E)
    # Note: We assume ids are in range [0, E). If not, this would be incorrect, but get_inputs
    # produces valid indices. Masking ensures we don't add for out-of-range masked elements.
    for i in range(0, BLOCK):
        if mask[i]:
            idx = ids[i]
            tl.atomic_add(counts_ptr + idx, 1)

@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive scan: offsets[k+1] = offsets[k] + counts[k]
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mask = k < E
    prev = tl.load(offsets_ptr, mask=mask, other=0)
    cur = tl.load(counts_ptr + k, mask=mask, other=0)
    new = prev + cur
    tl.store(offsets_ptr + k, new, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx is (batch_size, seq_len, num_experts_per_tok), int32, on CUDA
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        BLOCK_SCAN = 256
        grid_scan = (triton.cdiv(E, BLOCK_SCAN),)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort permutation via torch for correctness
        # sorted_token_indices is the stable permutation of the flattened indices
        # Note: torch.argsort returns indices that would sort the tensor; we use stable=True for exact match
        sorted_token_indices = torch.argsort(x, stable=True).values  # indices in [0, N)

        return sorted_token_indices.to(torch.int32), offsets