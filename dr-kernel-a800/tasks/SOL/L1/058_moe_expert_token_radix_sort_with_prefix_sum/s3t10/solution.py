import torch
import triton
import triton.language as tl


@triton.jit
def bitonic_bit_sort(flat_ptr, N, out_idx_ptr, bit_pos: tl.constexpr):
    """
    Perform one step of counting-sort-based bitonic network for index array 'out_idx_ptr'
    using the 'bit' at position 'bit_pos' (0 <= bit_pos < 8 since num_experts=256=2^8).
    This kernel sorts the array out_idx_ptr (of length N) by the bit at bit_pos, and
    because we run it from LSB to MSB, the final order will be stable.
    Implementation details:
    - counts: number of elements with ((flat[out_idx_ptr[i]] >> bit_pos) & 1) == 1
    - exclusive prefix sum to get positions
    - scatter copy to out_idx_ptr[position[i]] = i
    """
    # counts of ones at this bit position among values referenced by out_idx_ptr
    ones_count = tl.zeros((), dtype=tl.int32)
    for i in range(N):
        idx = tl.load(out_idx_ptr + i)
        val = tl.load(flat_ptr + idx)
        bit = (val >> bit_pos) & 1
        ones_count += tl.where(bit == 1, 1, 0)

    # exclusive prefix sum for position calculation
    positions = tl.zeros((N,), dtype=tl.int32)
    # Compute exclusive prefix sum for position mapping
    # For elements with bit==1, position = exclusive_prefix_sum(ones_count)
    # For elements with bit==0, position = exclusive_prefix_sum(cumulative0)
    # We need to track cumulative0 and cumulative1.
    # To do this efficiently, we recompute cumulatives using scans over N with masks.
    # This is O(N) per bit and acceptable for N up to a few thousand.
    # Implementation: loop over i to compute positions using masks.

    # Prepare masks for ones and zeros
    for i in range(N):
        idx = tl.load(out_idx_ptr + i)
        val = tl.load(flat_ptr + idx)
        bit_i = (val >> bit_pos) & 1

        # Compute cumulative1 and cumulative0 for i:
        # cumulative1_up to i-1: sum_{k=0..i-1} ones_count[k]
        # but ones_count is scalar; instead, we use scalar and index mapping trick.
        # Since ones_count is scalar, we compute positions with simple mapping:
        # If bit_i == 1, position = sum_{j < i: bit_j == 1}
        # If bit_i == 0, position = sum_{j < i: bit_j == 0}
        # We can compute with a second loop that counts how many elements before i share the same bit.
        count_before_i_one = tl.zeros((), dtype=tl.int32)
        count_before_i_zero = tl.zeros((), dtype=tl.int32)
        for j in range(i):
            idxj = tl.load(out_idx_ptr + j)
            valj = tl.load(flat_ptr + idxj)
            bitj = (valj >> bit_pos) & 1
            count_before_i_one += tl.where(bitj == 1, 1, 0)
            count_before_i_zero += tl.where(bitj == 0, 1, 0)

        # Final position for i:
        pos_i = tl.where(bit_i == 1, count_before_i_one, count_before_i_zero)
        positions[i] = pos_i

    # Now scatter copy: write original index i into out_idx_ptr[positions[i]]
    for i in range(N):
        pos_i = positions[i]
        tl.store(out_idx_ptr + pos_i, tl.load(out_idx_ptr + i))


@triton.jit
def count_histogram_atomic(flat_ptr, N, counts_ptr, BLOCK: tl.constexpr):
    """
    Parallel histogram using atomic adds:
    For each element i in [0, N), if i % BLOCK < N:
      For expert e in 0..255:
        if flat[i] == e: atomic_add counts[e] by 1
    This ensures we count all elements exactly once regardless of grid size.
    Here, grid=(1,) and BLOCK=N to cover all elements.
    """
    pid = tl.program_id(axis=0)
    for i in range(N):
        val = tl.load(flat_ptr + i)
        for e in range(256):
            if val == e:
                tl.atomic_add(counts_ptr + e, 1)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts into offsets_ptr[0..N_bins-1]:
    offsets[i] = sum_{k=0..i-1} counts[k] for i in 1..N_bins-1; offsets[0] = 0.
    This is O(N_bins^2), acceptable for N_bins=256.
    """
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    for i in range(1, N_bins):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    """
    Triton-optimized version of the original Model:
    - No torch.sort or torch.bincount in forward; all computations in Triton kernels.
    - sorted_token_indices: permutation of original flattened indices that sorts values ascending stably.
      We implement a Triton counting-sort-based bitonic network over 8 bit positions.
    - expert_offsets: exclusive prefix sum over histogram of expert indices computed in Triton.
    """
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA for Triton
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to('cuda')

        # Flatten and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton argsort via bitonic counting sort per bit (stable since LSB->MSB)
        # Start with out_idx_ptr as original indices 0..N-1
        out_idx = torch.arange(N, dtype=torch.int32, device=device)

        # Run 8 bit positions to fully sort
        for bit_pos in range(8):
            bitonic_bit_sort[(1,)](flat, N, out_idx, bit_pos=bit_pos, num_warps=4)

        # After 8 iterations, out_idx should be the stable argsort permutation of flat.
        sorted_token_indices = out_idx

        # 2) Triton histogram: counts per expert
        counts = torch.empty(256, dtype=torch.int32, device=device)
        count_histogram_atomic[(1,)](flat, N, counts, BLOCK=N, num_warps=4)

        # 3) Triton exclusive prefix sum to produce expert_offsets (length = num_experts + 1)
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=256, num_warps=4)

        # Return results: (sorted_token_indices, expert_offsets)
        return sorted_token_indices, offsets


# Helper to generate inputs (optional, not required by harness):
def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
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


# Example usage:
# model = ModelNew().cuda()
# inputs = get_inputs({"batch_size": 8, "seq_len": 256, "num_experts": 256, "num_experts_per_tok": 4}, device="cuda")
# topk_idx = inputs["topk_idx"]
# sorted_idx, offsets = model(topk_idx)
# print(sorted_idx.shape, offsets.shape)


def run(*args):
    return ModelNew()(*args)
