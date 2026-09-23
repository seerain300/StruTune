import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements and atomically increments counts[e] for each occurrence of e
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load a block of indices; cast to int32 for modulo/division
    vals = tl.load(x_ptr + offs, mask=mask, other=0)
    vals = vals.to(tl.int32)
    # Process up to BLOCK elements per program; dynamic loop over i is fine here
    for i in range(BLOCK):
        # guard: only process valid lanes
        valid_i = mask[i]
        id_val = vals[i]
        # id_val may be out of range for invalid lanes; ensure we don't update counts for invalid
        if valid_i:
            # atomic add into counts[id_val]
            tl.atomic_add(counts_ptr + id_val, 1)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive prefix sums of counts into offsets[0..E]
    # Initialize offsets[0] on host, this kernel handles offsets[1..E]
    acc = tl.zeros((), dtype=tl.int32)
    for k in range(E):
        c = tl.load(counts_ptr + k)
        acc += c
        tl.store(offsets_ptr + k + 1, acc)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Produce sorted_token_indices = out where out[pos] = the original flattened index after stable sort
    # starts_ptr points to the exclusive prefix sums (offsets of each expert).
    for pos in range(0, N):
        id_val = tl.load(x_ptr + pos).to(tl.int32)
        idx = tl.load(starts_ptr + id_val).to(tl.int32)  # exclusive start index for id_val
        tl.store(out_ptr + idx, pos)
        tl.store(starts_ptr + id_val, idx + 1)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = int(num_experts)

    def forward(self, topk_idx: torch.Tensor):
        # Assume topk_idx is provided by get_inputs and is on CUDA device
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = self.num_experts

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 2048
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0  # inclusive prefix sum starts at 0
        BLOCK_SCAN = 256
        grid_scan = (1,)  # one program handles all E; inner loop is fine for E=256
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        BLOCK_SORT = 1
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=BLOCK_SORT)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets