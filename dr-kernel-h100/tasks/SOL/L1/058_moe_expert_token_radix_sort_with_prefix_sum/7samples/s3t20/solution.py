import torch
import triton
import triton.language as tl


@triton.jit
def _stable_argsort_indices_kernel(
    flat_ptr,         # *int32
    out_ptr,          # *int32, length N, will hold 0..N-1 in sorted order
    N,                # int32
):
    # One program per original index i
    i = tl.program_id(0)
    # Guard if grid > N
    if i >= N:
        return

    # Load value at position i
    v = tl.load(flat_ptr + i)
    # Compute stable rank:
    # rank = count of elements < v + count of elements == v with original index < i
    count_less = tl.zeros((), dtype=tl.int32)
    count_equal_before = tl.zeros((), dtype=tl.int32)

    # Compare against all j
    for j in range(0, N):
        val_j = tl.load(flat_ptr + j)
        # count elements strictly less than v
        count_less += (val_j < v).to(tl.int32)
        # count elements equal to v and have original index j < i (stable tie-break)
        count_equal_before += (val_j == v).to(tl.int32) * (j < i).to(tl.int32)

    rank = count_less + count_equal_before

    # Reserve a unique position via atomic_add
    pos = tl.atomic_add(out_ptr, 1)
    # Store original index i at position 'rank'
    tl.store(out_ptr + pos, i)


@triton.jit
def _histogram_kernel(
    flat_ptr,         # *int32
    N,                # int32
    histogram_ptr,    # *int32, length num_experts
    num_buckets: tl.constexpr,  # number of expert IDs (256)
):
    # Simple atomic histogram: one pass over N
    for idx in range(0, N):
        val = tl.load(flat_ptr + idx)
        # Only consider buckets in [0, num_buckets)
        tl.atomic_add(histogram_ptr + val, 1)


@triton.jit
def _inclusive_scan_prefix_sum(
    input_ptr,         # *int32, length num_experts
    output_ptr,        # *int32, length (num_experts + 1)
    num_buckets: tl.constexpr
):
    # Copy input -> offsets[1..] and compute inclusive scan in-place
    # We assume host has already set offsets[0] = 0.
    # Iterate up to 2*num_buckets steps (sufficient for 256).
    for step in range(1, 1 + num_buckets):
        # First copy
        tl.store(output_ptr + step, tl.load(input_ptr + (step - 1)))
        # Then scan
        for i in range(1, num_buckets):
            output_ptr[i] = output_ptr[i] + output_ptr[i - 1]


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of:
          - sorted_token_indices: 1D int32 tensor of length N = topk_idx.numel(), matching
                                   torch.argsort(topk_idx.reshape(-1), stable=True).indices
          - expert_offsets: 1D int32 tensor of length (num_experts + 1), where num_experts=256
        """
        # Flatten and ensure int32, contiguous
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        device = flat.device
        N = flat.numel()

        # 1) Stable argsort: permutation indices using Triton
        out = torch.zeros(N, dtype=torch.int32, device=device)  # positions holder
        grid = (triton.cdiv(N, 1),)  # one program per index
        _stable_argsort_indices_kernel[grid](flat, out, N)

        # 2) Histogram of expert IDs using Triton
        num_experts = 256  # matches original code
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(1,)](flat, N, histogram, num_buckets=num_experts)

        # 3) Inclusive prefix sum to get expert offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0  # ensure starts at 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_buckets=num_experts)

        return out, offsets


def run(*args):
    return ModelNew()(*args)
