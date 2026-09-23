import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements of x, atomically incrementing counts
    pid = tl.program_id(0)
    start = pid * BLOCK
    # Vector of indices for this program
    offs = start + tl.arange(0, BLOCK)
    mask = offs < N
    # Load values; if out-of-range, set to -1 (unused)
    vals = tl.load(x_ptr + offs, mask=mask, other=-1)
    # Atomic add 1 for each valid element to its expert count
    # Note: We can't index into counts with a vector, so loop over valid lanes
    # Triton allows python for-loops with runtime bounds when controlling by mask
    for i in range(BLOCK):
        if mask[i]:
            val = vals[i]  # scalar int32
            # Guard val in [0, E-1]
            # Triton supports int32 atomics; E is constexpr at compile-time per launch
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Compute inclusive prefix sums: offsets[i] = sum_{j<=i} counts[j], for i in [0..E-1]
    # Initialize offsets[0] = 0 on host before calling
    # Simple sequential scan per element to avoid complex block logic
    for i in range(E):
        # offsets_ptr[i] = offsets_ptr[i-1] + counts_ptr[i]
        if i == 0:
            tl.store(offsets_ptr + i, tl.load(offsets_ptr + i))  # keep 0
        else:
            prev = tl.load(offsets_ptr + (i - 1))
            curr_count = tl.load(counts_ptr + i)
            tl.store(offsets_ptr + i, prev + curr_count)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: produces permutation sorted_token_indices
    # For each position pos in [0, N), id = x[pos], write out[starts[id]] = pos, then starts[id] += 1
    # Simple sequential loop; BLOCK=1 covers all pos safely for typical sizes.
    for pos in range(N):
        id = tl.load(x_ptr + pos)
        # Load current start and write out pos
        cur_start = tl.load(starts_ptr + id)
        tl.store(out_ptr + cur_start, pos)
        # Update start
        tl.atomic_add(starts_ptr + id, 1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        # Ensure we use Triton-only; no torch ops here
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing
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
        offsets[0] = 0  # inclusive prefix sum will fill offsets[1..E]
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
