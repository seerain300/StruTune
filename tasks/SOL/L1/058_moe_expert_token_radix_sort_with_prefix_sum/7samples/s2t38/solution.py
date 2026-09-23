import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    # Process a block of elements
    for offset in range(BLOCK):
        i = start + offset
        in_bounds = i < N
        # Load x[i] as int32; mask prevents out-of-bounds access
        x_val = tl.load(x_ptr + i, mask=in_bounds, other=0)  # int32
        # Compute expert id: x_val % E. E is scalar int.
        id = x_val % E
        # Atomic add into counts[id]
        tl.atomic_add(counts_ptr + id, 1, mask=in_bounds)


@triton.jit
def inclusive_scan_prefixsum(counts_ptr, offsets_ptr, E, BLOCK: tl.constexpr):
    # Simple per-element inclusive scan; E is small (e.g., 256), so BLOCK=1 is fine.
    for k in range(E):
        v = tl.load(counts_ptr + k)
        tl.store(offsets_ptr + k + 1, v + tl.load(offsets_ptr + k))


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E, BLOCK: tl.constexpr):
    # Stable sort: for each position pos, write out[starts[id]] = pos, then starts[id] += 1
    for pos in range(N):
        x_val = tl.load(x_ptr + pos)
        id = x_val % E
        # Write current position into output at starts[id]
        tl.store(out_ptr + tl.load(starts_ptr + id), pos)
        # Increment starts[id] (exclusive prefix)
        inc = 1
        tl.store(starts_ptr + id, tl.load(starts_ptr + id) + inc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # topk_idx is the tensor provided by get_inputs: int32, on CUDA device
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"

        # Flatten to 1D
        x = topk_idx.reshape(-1)  # int32
        N = x.numel()

        # num_experts is expected to be provided by the harness; assume E=256 as in original
        E = 256

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK_HIST = 1024
        grid_hist = (triton.cdiv(N, BLOCK_HIST),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK_HIST)

        # 2) Inclusive prefix sum to produce expert_offsets (length E+1)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0  # inclusive scan will fill subsequent offsets
        BLOCK_SCAN = 1
        grid_scan = (1,)
        inclusive_scan_prefixsum[grid_scan](counts, offsets, E, BLOCK=BLOCK_SCAN)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation of 0..N-1)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E, BLOCK=1)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
