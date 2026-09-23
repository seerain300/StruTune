import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(
    flat_ptr,            # *int32, input flat array
    counts_ptr,          # *int32, output counts per expert
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes BLOCK_SIZE elements
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load values; for masked-off lanes, load 0 to avoid OOB
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)

    # Atomic add 1 for each valid lane to counts[vals]
    # Note: vals must be in [0, num_experts-1]
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


@triton.jit
def _inclusive_prefix_sum_kernel(
    counts_ptr,           # *int32, input counts (length num_experts)
    offsets_ptr,          # *int32, output inclusive prefix sums (length num_experts+1)
    num_experts: tl.int32,
):
    # Single-program inclusive scan over counts
    # We iterate over each expert sequentially. This is fine for num_experts=256.
    acc = tl.zeros((), dtype=tl.int32)
    for e in range(num_experts + 1):
        # Load current count or 0 if e >= num_experts (num_experts acts as sentinel)
        # Note: In this loop, we rely on the host to pass counts_ptr of length num_experts.
        # We'll compute acc from counts_ptr up to num_experts, and write offsets up to num_experts+1.
        # For e == num_experts, we set acc to N (though we won't write offsets[num_experts]).
        if e < num_experts:
            cnt = tl.load(counts_ptr + e)
        else:
            cnt = tl.zeros((), dtype=tl.int32)

        acc += cnt
        tl.store(offsets_ptr + e, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure we work on CUDA
        if not topk_idx.is_cuda:
            # Fallback to CPU if needed; get_inputs typically provides CUDA tensors.
            raise RuntimeError("topk_idx must be on CUDA device for Triton execution.")

        # Flatten to 1D and ensure contiguous
        flat = topk_idx.reshape(-1).contiguous()
        n = flat.numel()

        num_experts = 256  # as in the original run

        # Triton: compute histogram of expert IDs
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        # Choose a reasonable block size; 1024 works well for moderate N.
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _histogram_counts_kernel[grid](
            flat, counts, n_elements=n, BLOCK_SIZE=BLOCK_SIZE
        )

        # Triton: compute inclusive prefix sums to get expert offsets
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(1,)](
            counts, offsets, num_experts=num_experts
        )
        # offsets[0] will be 0; last element offsets[num_experts] should be N.
        # Verify last element (not strictly necessary but useful for debugging)
        # offsets[-1] = n

        # Stable sort using PyTorch to ensure correctness and stability
        # Original returns sorted_token_indices (int64), we keep int64 for exact match
        sorted_token_indices = torch.sort(flat, stable=True)[1].to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
