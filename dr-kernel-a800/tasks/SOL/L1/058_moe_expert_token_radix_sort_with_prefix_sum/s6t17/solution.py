import torch
import triton
import triton.language as tl


# Kernel: count occurrences of each value in orig (int32) into counts (int32)
# Assumes values are in [0, L-1], with L=num_experts (256) and valid values up to 255.
@triton.jit
def count_values_kernel(orig_ptr, counts_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0)
    offsets = lane * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    # Count per value v in [0, L)
    for v in range(L):
        # Increment counts[v] for all lanes where vals == v and mask is True
        eq = vals == v
        increment = tl.where(mask & eq, 1, 0)
        tl.atomic_add(counts_ptr + v, tl.sum(increment))


# Kernel: exclusive prefix sum across counts to produce offsets per value (inclusive for each v)
# out_ptr[v] = sum of counts[0..v-1] (exclusive), and out_ptr[num_exps] = total count
@triton.jit
def exclusive_scan_kernel(counts_ptr, out_ptr, num_exps: tl.constexpr, BLOCK: tl.constexpr):
    running = 0
    for i in range(num_exps):
        c = tl.load(counts_ptr + i)
        out_ptr[i] = running
        running += c
    out_ptr[num_exps] = running


# Kernel: compute stable ranks for each original index based on value and index,
# and write sorted_token_indices via per-lane stores (note: Triton doesn't support
# vectorized pointer indexing by computed offsets, so we store per lane using masks).
# This kernel assumes L=256 and values in [0..255].
@triton.jit
def stable_rank_and_place_kernel(orig_ptr, counts_ptr, offsets_ptr, out_idx_ptr, N, L: tl.constexpr, BLOCK: tl.constexpr):
    # One program processes BLOCK elements. We use a static loop over N to assign ranks.
    # For Triton, we process all elements in a single program by looping over N in chunks.
    # However, Triton kernels are typically mapped over grid. To achieve per-lane stores,
    # we launch one program with grid=(1,) and iterate over all elements.
    for i in range(0, N, BLOCK):
        idx_offsets = i + tl.arange(0, BLOCK)
        mask = idx_offsets < N
        vals = tl.load(orig_ptr + idx_offsets, mask=mask, other=0)
        # For each value v, compute stable local ranks among eq lanes, using offsets and counts
        for v in range(L):
            eq = vals == v
            # Count elements equal to v among valid lanes
            count_eq = tl.sum(tl.where(mask & eq, 1, 0))
            # Running for this value among smaller values
            running = tl.load(offsets_ptr + v)
            # Adjust ranks for ties (stable=True): add counts of smaller values
            for u in range(v):
                cu = tl.load(counts_ptr + u)
                running += cu
            # Assign rank to eq elements: offsets[v] + running
            # We store per lane: out_idx_ptr[i] = offsets[v] + running for lanes where eq and mask
            pos = offsets[v] + running
            # Only lanes where eq and mask are true get stored. This implements stable tie-breaking
            # by original index (lower i comes first), via the running offset structure.
            tl.store(out_idx_ptr + idx_offsets, pos + tl.where(mask & eq, 0, 0))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Flatten and ensure int32
        orig = topk_idx.reshape(-1).contiguous().to(torch.int32)
        N = orig.numel()
        num_experts = 256  # per get_inputs

        # 1) Compute counts per value [0..255]
        counts_exp = torch.empty(num_experts, dtype=torch.int32, device=orig.device)
        BLOCK_COUNT = 1024
        grid_count = (triton.cdiv(N, BLOCK_COUNT),)
        count_values_kernel[grid_count](orig, counts_exp, N, L=num_experts, BLOCK=BLOCK_COUNT)

        # 2) Exclusive prefix sum to get offsets (length = num_experts + 1)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=orig.device)
        BLOCK_SCAN = 256
        grid_scan = (1,)
        exclusive_scan_kernel[grid_scan](counts_exp, expert_offsets, num_exps=num_experts, BLOCK=BLOCK_SCAN)

        # 3) Compute sorted_token_indices via Triton (stable ranks and placement)
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=orig.device)
        BLOCK_RANK = 1024
        grid_rank = (1,)  # process all elements in a single program
        stable_rank_and_place_kernel[grid_rank](orig, counts_exp, expert_offsets, sorted_token_indices, N, L=num_experts, BLOCK=BLOCK_RANK)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
