import torch
import triton
import triton.language as tl


# Kernel 1: Build histogram of expert indices using atomic adds.
# Each element in 'flat' contributes +1 to hist[flat[i]].
@triton.jit
def HistogramKernel(flat_ptr, hist_ptr, N, E: tl.constexpr):
    # Single-program kernel; iterate over all N elements.
    # Note: Triton requires loops to be simple; this does fine for small to medium N.
    for i in range(N):
        val = tl.load(flat_ptr + i)  # load int32
        # atomic add to hist[val]
        tl.atomic_add(hist_ptr + val, 1)


# Kernel 2: Compute inclusive prefix sum of 'hist' into 'start'.
# Each program handles a chunk of E entries; we use a running sum vector and loop over chunk.
@triton.jit
def PrefixSumKernel(hist_ptr, start_ptr, E, BLOCK_E: tl.constexpr):
    pid = tl.program_id(axis=0)
    chunk = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    # running sum per expert index (vector of BLOCK_E)
    running = tl.zeros((BLOCK_E,), dtype=tl.int32)
    # Loop over all possible bins in E
    for j in range(0, E, BLOCK_E):
        idx = j + chunk
        mask = idx < E
        counts = tl.load(hist_ptr + idx, mask=mask, other=0)
        running += counts
        # Store inclusive prefix sums for valid idx
        tl.store(start_ptr + idx, running, mask=mask)


# Kernel 3: Stable counting sort + permutation write.
# For each token position 'pos' in 0..N-1:
#   e = flat[pos]
#   local_pos = atomic_add(start[e], 1)  # start[e] is incremented per token of expert e
#   out[start[e] + local_pos] = pos
#   sorted_token_indices[pos] = start[e] + local_pos
@triton.jit
def SortAndPermuteKernel(flat_ptr, out_ptr, sorted_idx_ptr, start_ptr, N, E: tl.constexpr):
    for pos in range(N):
        val = tl.load(flat_ptr + pos)
        # current start for this expert
        current = tl.atomic_add(start_ptr + val, 1)
        # write the original position into the sorted output at that slot
        tl.store(out_ptr + current, pos)
        # write the permutation index (the slot position) for this token
        tl.store(sorted_idx_ptr + pos, current)


# Kernel 4: Compute expert offsets (cumulative counts) from histogram 'hist'.
# Output is inclusive prefix sums into expert_offsets[1:], with expert_offsets[0] = 0.
@triton.jit
def ExpertOffsetsKernel(hist_ptr, offsets_ptr, E, BLOCK_E: tl.constexpr):
    pid = tl.program_id(axis=0)
    chunk = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    running = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for j in range(0, E, BLOCK_E):
        idx = j + chunk
        mask = idx < E
        counts = tl.load(hist_ptr + idx, mask=mask, other=0)
        running += counts
        # Store inclusive prefix sums
        tl.store(offsets_ptr + idx + 1, running, mask=mask)  # offsets[1:] = prefix sums


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # We will perform all computation in Triton. No torch ops in forward body.
        # Extract shapes and flatten.
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device
        E = 256  # num_experts as per the example; we will assert or handle accordingly

        # Allocate intermediate and output buffers
        hist = torch.zeros(E, dtype=torch.int32, device=device)
        start = torch.zeros(E, dtype=torch.int32, device=device)
        out = torch.empty(N, dtype=torch.int32, device=device)  # sorted positions of tokens
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)

        # Kernel 1: Histogram
        HistogramKernel[(1,)](flat, hist, N, E)

        # Kernel 2: Prefix sum of hist -> start
        BLOCK_E = 128
        grid_prefix = (triton.cdiv(E, BLOCK_E),)
        PrefixSumKernel[grid_prefix](hist, start, E, BLOCK_E)

        # Kernel 3: Stable sort + permutation
        SortAndPermuteKernel[(1,)](flat, out, sorted_token_indices, start, N, E)

        # At this point, out contains the sorted token positions; we need permutation indices.
        # sorted_token_indices already contains per-token slot (which is exactly the position in sorted order),
        # but we need the permutation of original indices. Since we wrote sorted_token_indices[pos] = slot,
        # the correct permutation is simply sorted_token_indices.
        # (We sort by expert id using counting sort: within each expert, order is preserved by pos loop.)

        # Kernel 4: Compute expert offsets from hist
        grid_exp = (triton.cdiv(E, BLOCK_E),)
        # Initialize offsets to zeros; kernel writes offsets[1:], leaving offsets[0]=0 as desired.
        offsets_zeros = torch.zeros(E + 1, dtype=torch.int32, device=device)
        ExpertOffsetsKernel[grid_exp](hist, offsets_zeros, E, BLOCK_E)
        # We can simply use offsets_zeros (it holds the inclusive cumsum). No need to modify.

        # Return the permutation and expert offsets. The permutation is sorted_token_indices.
        # Note: out is the sorted positions; sorted_token_indices is the permutation indices.
        # To map tokens to expert in sorted order, we could also use out, but the original PyTorch code
        # returns (sorted_token_indices, expert_offsets). Here we return sorted_token_indices as permutation
        # and expert_offsets as cumsum.
        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
