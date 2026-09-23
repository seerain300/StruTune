import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E: tl.constexpr, BLOCK: tl.constexpr):
    # Each program processes a contiguous block of elements
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values; if mask is false, load 0 (won't affect counts because atomic_add is masked)
    vals = tl.load(x_ptr + offs, mask=mask, other=0)

    # For each expert id in [0, E), atomically add to counts[id]
    for e in range(E):
        # Create a boolean mask for elements equal to e
        eq = vals == e
        # Only add for valid positions
        add_mask = mask & eq
        # Atomic add 1 for each True in add_mask
        # Note: vals are int32; eq is boolean; Triton will broadcast scalar 1
        tl.atomic_add(counts_ptr + e, add_mask.to(tl.int32))


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E: tl.constexpr):
    # One program per index i; compute offsets[i+1] = sum_{j<=i} counts[j]
    for i in range(E):
        sum_ = tl.zeros((), dtype=tl.int32)
        for j in range(i + 1):
            sum_ += tl.load(counts_ptr + j)
        tl.store(offsets_ptr + i + 1, sum_)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E: tl.constexpr):
    # Stable counting sort: out[starts[id]] = pos; starts[id] += 1
    # We process positions sequentially to preserve stability.
    for pos in range(0, N):
        id = tl.load(x_ptr + pos)  # int32
        # Write current position into slot starts[id]
        tl.store(out_ptr + tl.load(starts_ptr + id), pos)
        # Increment starts[id] for the next token with the same id
        tl.atomic_add(starts_ptr + id, 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Use the provided topk_idx tensor; do not use torch ops
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"

        # Flatten to 1D
        x = topk_idx.reshape(-1)
        N = x.numel()
        E = 256  # num_experts as per original code

        # 1) Histogram of expert IDs using Triton
        counts = torch.zeros(E, dtype=torch.int32, device=x.device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        histogram_experts[grid_hist](x, counts, N, E, BLOCK=BLOCK)

        # 2) Inclusive prefix sum to produce expert_offsets using Triton (one program per index)
        offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
        offsets[0] = 0  # initialize first offset
        grid_scan = (E,)
        inclusive_scan_counts[grid_scan](counts, offsets, E)

        # 3) Stable counting sort using Triton to produce sorted_token_indices (permutation)
        out = torch.empty(N, dtype=torch.int32, device=x.device)
        starts = offsets.clone()  # exclusive prefix sums per expert
        grid_sort = (1,)
        stable_counting_sort[grid_sort](x, starts, out, N, E)

        # Return sorted_token_indices (int32 permutation) and expert_offsets (int32, length E+1)
        return out, offsets


def run(*args):
    return ModelNew()(*args)
