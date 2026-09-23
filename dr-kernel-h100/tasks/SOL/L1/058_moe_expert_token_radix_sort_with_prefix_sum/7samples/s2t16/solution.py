import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load expert IDs from x (int32)
    ids = tl.load(x_ptr + offs, mask=mask, other=0)
    # Compute expert index for atomic add: e = ids % E
    e = ids % E
    # Mask invalid lanes
    mask2 = mask & (e >= 0) & (e < E)
    # Atomic add into counts
    tl.atomic_add(counts_ptr + e, 1, mask=mask2)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Single-program inclusive scan over E elements
    # offsets[0] must be initialized to 0 on host
    # We process in chunks of BLOCK and loop within the kernel
    # This is simple and correct for small E (like 256)
    acc = tl.zeros((), dtype=tl.int32)
    # Loop over elements 0..E-1
    for i in range(0, E):
        # Load count
        c = tl.load(counts_ptr + i)
        # Update accumulator
        acc += c
        # Write inclusive prefix sum
        tl.store(offsets_ptr + i + 1, acc)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: produce permutation 'out' such that x[out[k]] is sorted by expert IDs.
    # We iterate positions in ascending order to ensure stability for ties.
    # For each position pos, read id = x[pos], write out[starts[id]] = pos, then starts[id] += 1.
    acc = tl.zeros((), dtype=tl.int32)
    for pos in range(0, N):
        # Load id
        id_val = tl.load(x_ptr + pos)
        # Compute current start for this expert
        start = tl.load(starts_ptr + id_val)
        # Store the original position
        tl.store(out_ptr + start, pos)
        # Advance start
        new_start = start + 1
        # Atomic update starts[id] = new_start
        tl.atomic_add(starts_ptr + id_val, 1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # The evaluation harness provides topk_idx as the first (and only) argument.
        # Ensure it is a CUDA tensor and int32.
        assert len(args) == 1, "forward expects one input tensor"
        topk_idx = args[0]
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = 256  # num_experts as per original code

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        grid_scan = (1,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=1)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
