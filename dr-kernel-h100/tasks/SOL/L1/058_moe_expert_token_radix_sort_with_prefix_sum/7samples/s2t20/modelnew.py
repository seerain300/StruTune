import torch
import triton
import triton.language as tl


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program processes BLOCK elements, counting occurrences of each expert id
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load a block of x
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Compute expert id for each element as x % E
    # Note: x is int32; E is int32 scalar
    id = x % E

    # Atomically add 1 to counts[id] for valid elements
    # Mask ensures we only count valid positions
    tl.atomic_add(counts_ptr + id, 1, mask=mask)


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Compute inclusive prefix sums: offsets[i+1] = offsets[i] + counts[i]
    # We run one program per index i; offsets[0] must be initialized on host.
    i = tl.program_id(0)
    if i < E:
        # Initialize current with counts[i]
        current = tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, current)
        # Loop over j > i to update suffix sums
        # We only update positions after i
        # Note: Triton for-loops need compile-time constants; we iterate up to E-1 and guard updates.
        for j in range(0, E):  # unrolled loop; valid since E is small (256)
            if j > i:
                prev = tl.load(offsets_ptr + j)
                new = prev + current
                tl.store(offsets_ptr + j, new)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable counting sort: iterate positions, write out[starts[id]] = pos, then starts[id] += 1
    pid = tl.program_id(0)
    if pid == 0:
        for pos in range(0, N):
            id = tl.load(x_ptr + pos)
            # Stable tie-breaking by position: we store in ascending order
            idx = tl.load(starts_ptr + id)
            tl.store(out_ptr + idx, pos)
            tl.atomic_add(starts_ptr + id, 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Use provided topk_idx exactly; do not generate or alter with torch ops
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D for processing
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = 256  # num_experts as in the original run

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 2048  # larger block improves throughput
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0
        # One program per element to compute prefix sums sequentially; E is small (256), so overhead is low
        grid_scan = (E,)
        inclusive_scan_counts[grid_scan](counts, offsets, E, BLOCK=1)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums for each expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets