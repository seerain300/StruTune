import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, N: tl.int32, num_experts: tl.int32, BLOCK_H: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_H
    offsets = block_start + tl.arange(0, BLOCK_H)
    mask = offsets < N
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0).to(tl.int32)
    for i in range(BLOCK_H):
        idx = offsets[i]
        if mask[i]:
            v = vals[i]
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def inclusive_scan_inplace(counts_ptr, E: tl.int32, LOG_E: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid < E:
        x = tl.load(counts_ptr + pid)
        for i in range(LOG_E):
            j = pid ^ i
            if j < pid:
                carry = tl.load(counts_ptr + j)
                x += carry
                tl.store(counts_ptr + pid, x)


@triton.jit
def selection_sort_indices(vals_ptr, idx_out_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Two-pass selection sort: produce argsort indices
    # We repeatedly find the minimum among remaining elements and write its index to idx_out[round].
    # This is O(N^2); acceptable for correctness demonstration.
    for round in range(0, BLOCK):  # loop up to N rounds
        # If round >= N, we can break; Triton doesn't support early breaks easily.
        # We emulate by masking and setting INF when idx is >= N.
        INF = (1 << 31) - 1
        min_val = INF
        min_idx = 0
        # Find minimum value among valid idx positions
        for i in range(BLOCK):
            idx = i
            # Load value at idx; if idx >= N, value=INF
            val = tl.load(vals_ptr + idx, mask=(idx < N), other=INF)
            is_new_min = val < min_val
            # Update min_val and min_idx
            min_val = tl.where(is_new_min, val, min_val)
            min_idx = tl.where(is_new_min, idx, min_idx)
        # Write min_idx to sorted position 'round'
        tl.store(idx_out_ptr + round, min_idx)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        device = topk_idx.device
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"

        # Flatten
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Histogram in Triton
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        BLOCK_H = 1024
        grid_h = triton.cdiv(N, BLOCK_H)
        histogram_kernel[(grid_h,)](flat, counts, N, num_experts, BLOCK_H)

        # 2) Inclusive scan to get expert_offsets
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        expert_offsets[1:] = counts
        LOG_E = 8  # log2(256)
        inclusive_scan_inplace[(num_experts,)](expert_offsets, num_experts, LOG_E)

        # 3) Selection sort in Triton to produce argsort permutation
        # We sort up to N rounds; BLOCK must be >= N and a constexpr.
        # For safety, use BLOCK=4096; loop will be masked by N.
        BLOCK_SORT = 4096
        idx_out = torch.empty(BLOCK_SORT, dtype=torch.int32, device=device)
        selection_sort_indices[(1,)](flat, idx_out, N, BLOCK_SORT)

        # Extract first N sorted indices
        sorted_token_indices = idx_out[:N]

        return sorted_token_indices.to(torch.int32), expert_offsets


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    return {"topk_idx": topk_idx}


def run(*args):
    return ModelNew()(*args)
