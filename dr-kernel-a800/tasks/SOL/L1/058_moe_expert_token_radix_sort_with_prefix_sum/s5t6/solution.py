import torch
import triton
import triton.language as tl


# Kernel 1: Compute per-expert counts (histogram) for flat.
# Each program handles one expert k and scans the entire flat array in chunks to count matches.
@triton.jit
def _histogram_kernel(flat_ptr, counts_out_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(axis=0)
    if k >= NUM_EXPERTS:
        return
    cnt = tl.zeros((), dtype=tl.int32)
    for offs in range(0, M, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < M
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)
        cnt += tl.sum((vals == k) & mask, axis=0).to(tl.int32)
    tl.store(counts_out_ptr + k, cnt)


# Kernel 2: Compute inclusive prefix sums of counts -> offsets_incl[0..NUM_EXPERTS-1]
@triton.jit
def _inclusive_scan_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < NUM_EXPERTS
        vals = tl.load(counts_ptr + idx, mask=mask, other=0)
        blk_sum = tl.sum(vals, axis=0).to(tl.int32)
        acc += blk_sum
        tl.store(offsets_ptr + idx, acc, mask=mask)


# Kernel 3: Reduce sum of counts into a single scalar (total_count[0]), used to set expert_offsets[NUM_EXPERTS]
@triton.jit
def _reduce_sum_kernel(counts_ptr, out_ptr, N: tl.int32, BLOCK: tl.constexpr):
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, N, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < N
        vals = tl.load(counts_ptr + idx, mask=mask, other=0)
        acc += tl.sum(vals, axis=0).to(tl.int32)
    tl.store(out_ptr, acc)


# Kernel 4: Stable permutation: for each j, compute rank = base_excl + tie_count
@triton.jit
def _stable_permutation_kernel(flat_ptr, offsets_ptr, out_perm_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    for j in range(0, M):
        val_j = tl.load(flat_ptr + j)
        key_j = val_j
        if key_j == 0:
            base_excl = tl.zeros((), dtype=tl.int32)
        else:
            base_excl = tl.load(offsets_ptr + key_j - 1)
        # tie_count = number of earlier elements t < j with same key and smaller value
        tie_count = tl.zeros((), dtype=tl.int32)
        for t in range(0, j):
            val_t = tl.load(flat_ptr + t)
            key_t = val_t
            if key_t == key_j and val_t < val_j:
                tie_count += 1
        rank = base_excl + tie_count
        tl.store(out_perm_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts
        # Tunable block sizes for loops
        self._block_hist = 1024
        self._block_scan = 1024

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized implementation:
        - Computes sorted_token_indices: permutation of [0..M-1] ordered by values in flat (stable).
        - Computes expert_offsets: length = num_experts + 1, inclusive cumulative counts per expert + 1.
        Returns: (sorted_token_indices, expert_offsets)
        """
        # Ensure flat is CUDA and int32
        flat = topk_idx.reshape(-1)
        if not flat.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors for Triton kernels.")
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)
        device = flat.device
        M = flat.numel()
        NUM_EXPERTS = self.num_experts

        # 1) Compute counts per expert using Triton
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        grid_hist = (NUM_EXPERTS,)
        _histogram_kernel[grid_hist](flat, counts, M, NUM_EXPERTS, self._block_hist)

        # 2) Compute inclusive prefix sums (base positions) using Triton
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        grid_scan = (triton.cdiv(NUM_EXPERTS, self._block_scan),)
        _inclusive_scan_kernel[grid_scan](counts, offsets_incl, NUM_EXPERTS, self._block_scan)

        # 3) Compute total count for expert_offsets: counts.sum() (Triton reduction)
        total_count = torch.empty(1, dtype=torch.int32, device=device)
        _reduce_sum_kernel[(1,)](counts, total_count, NUM_EXPERTS, 1024)

        # 4) Prepare expert_offsets: inclusive prefix sums plus final +1
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        expert_offsets[:NUM_EXPERTS] = offsets_incl
        expert_offsets[NUM_EXPERTS] = total_count[0] + 1

        # 5) Compute stable sorted permutation indices using Triton
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        _stable_permutation_kernel[(M,)](flat, expert_offsets, sorted_token_indices, M, NUM_EXPERTS)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
