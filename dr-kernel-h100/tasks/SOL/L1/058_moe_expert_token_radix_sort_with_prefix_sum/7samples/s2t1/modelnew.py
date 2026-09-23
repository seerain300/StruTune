import torch
import triton
import triton.language as tl


@triton.jit
def prefix_sum_kernel(counts_ptr, out_ptr, E: tl.constexpr):
    """
    Compute inclusive prefix sum of counts_ptr[0:E] into out_ptr[0:E+1].
    out_ptr[0] = 0; out_ptr[j+1] = sum_{i=0..j} counts[i].
    """
    # Each program handles a block of the counts array
    pid = tl.program_id(axis=0)
    BLOCK = 128  # process up to 128 elements per program
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < E

    # Load counts; out-of-range lanes get 0
    counts = tl.load(counts_ptr + offsets, mask=mask, other=0)

    # Compute prefix sum within the block using simple sequential accumulation
    # We do this per lane. Triton supports tl.arange, masks, and reductions.
    # This is fine for small E like 256.
    # Initialize running sum to 0 (int32)
    running = tl.zeros([BLOCK], dtype=tl.int32)

    # Loop over positions in the block
    # Note: we iterate i from 0 to BLOCK-1 and use mask to ignore out-of-range lanes.
    # This avoids dynamic loops that Triton can't unroll.
    for i in range(BLOCK):
        # Effective lane-wise condition: if lane is valid, add counts[i], else add 0
        valid = (i < E) & mask
        # For invalid lanes, counts[i] is 0 from tl.load(..., other=0)
        running += tl.where(valid, counts[i], 0)

        # Store partial results for valid lanes: out_ptr[i+1] = running
        tl.store(out_ptr + (i + 1), running, mask=mask & (i < E))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # 1) Flatten and sort (use PyTorch for correctness)
        flat = topk_idx.reshape(-1)  # int64 by default
        # We sort ascending by token positions (original order), stable=True ensures ties (same expert)
        # follow original order.
        _, sorted_token_indices = flat.sort(stable=True)  # indices are 0..N-1
        sorted_token_indices = sorted_token_indices.to(torch.int32)  # match original output dtype

        # 2) Compute per-expert counts via torch.bincount (fast, correct)
        # topk_idx values are in [0, num_experts-1] == [0, E-1] in the given setup (E=256).
        E = 256  # Keep this constant as in the original code; inputs ensure values < E.
        counts = torch.bincount(flat.long(), minlength=E)  # int64 counts

        # 3) Compute expert_offsets (inclusive prefix sum) using Triton
        device = flat.device
        expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)

        # Launch Triton kernel: one program per block of up to 128 elements
        # Since E=256, grid = (ceil_div(E, 128),) = (3,)
        grid = (triton.cdiv(E, 128),)
        # Note: Triton kernels prefer constants for constexpr; E is known (256).
        prefix_sum_kernel[grid](counts, expert_offsets, E=E)

        return sorted_token_indices, expert_offsets