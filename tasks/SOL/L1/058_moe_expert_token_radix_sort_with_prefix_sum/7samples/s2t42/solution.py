import torch
import triton


@triton.jit
def histogram_experts(x_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    # Each program handles a block of elements; safely compute histogram with atomics
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    mask = idx < N

    # Load values; out-of-bounds masked elements are not dereferenced
    # Note: x_ptr is int32; Triton will treat it as such. We operate on a single vector.
    vals = tl.load(x_ptr + idx, mask=mask, other=0)
    # Reduce vector to scalar counts per element
    # We need to loop over the BLOCK vector to perform atomic adds. Use a simple while loop.
    i = 0
    while i < BLOCK:
        if mask[i]:
            v = vals[i]
            # Ensure v is within [0, E); histogram expects valid IDs
            # Triton supports int32 arithmetic, so cast to int32 if needed
            v = tl.cast(v, tl.int32)
            # Atomic add into counts
            tl.atomic_add(counts_ptr + v, 1)
        i += 1


@triton.jit
def inclusive_scan_counts(counts_ptr, offsets_ptr, E):
    # Inclusive prefix sum: offsets[i] = sum_{j <= i} counts[j], offsets[0] = 0
    total = 0
    for i in range(0, E):
        total += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i + 1, total)


@triton.jit
def stable_counting_sort(x_ptr, starts_ptr, out_ptr, N, E):
    # Stable sort via counting: out stores permutation, starts[e] points to next available slot
    # Initialize: starts is exclusive prefix sums; starts[e] = offsets[e], offsets[0]=0 handled outside
    pos = 0
    while pos < N:
        # Read id at position pos
        id = tl.load(x_ptr + pos)
        id = tl.cast(id, tl.int32)
        # Write current pos to out at starts[id], then advance starts[id]
        slot = tl.load(starts_ptr + id)
        tl.store(out_ptr + slot, pos)
        tl.atomic_add(starts_ptr + id, 1)  # next token for this expert goes to slot+1
        pos += 1


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        # topk_idx is the input tensor provided by get_inputs; must be used as-is (no torch ops to create/mutate)
        # Ensure CUDA for Triton execution
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        assert topk_idx.dtype == torch.int32, "topk_idx must be int32"

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
        offsets[0] = 0
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
