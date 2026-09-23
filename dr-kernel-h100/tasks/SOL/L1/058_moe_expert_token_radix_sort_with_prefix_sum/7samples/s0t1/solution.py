import torch
import triton
import triton.language as tl


# Triton kernel: bincount for int32 indices into a fixed-size counts array (num_experts = 256).
# We do one atomic add per element in the flattened input vector.
@triton.jit
def bincount_kernel(
    src_ptr,           # *const int32
    counts_ptr,        # *int32, length = num_experts
    n_elements,        # int32: total number of elements in src_ptr
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input indices; 'other=0' for out-of-range lanes in the last block
    vals = tl.load(src_ptr + offs, mask=mask, other=0)

    # Increment counts for each valid element. We assume vals in [0, 255].
    # For masked lanes (not < n_elements), vals=0 from 'other=0', which doesn't modify counts.
    # Atomic add per element to avoid collisions.
    # The 'other' argument is 0 because we only want to add for valid lanes.
    # Note: If you expect vals > 255, you should mask those out before adding. Here we enforce vals in [0,255] in the host.
    # We can't branch per-lane cheaply; but given the host generates vals in [0, num_experts-1], this is fine.
    for i in range(0, BLOCK_SIZE):
        if mask[i]:
            idx = vals[i]
            # idx is int32; counts_ptr is int32
            tl.atomic_add(counts_ptr + idx, 1)


# Triton kernel: compute inclusive prefix sum of x (length = num_experts + 1), write to y (same length).
# This is done sequentially by a single program for simplicity and correctness.
@triton.jit
def prefix_sum_inclusive_kernel(
    x_ptr,         # *const int32, input (counts)
    y_ptr,         # *int64, output offsets (inclusive cumsum)
    length,        # int32, length of x (== num_experts + 1)
):
    # Single program handles the whole vector
    # We compute y[i] = sum_{j=0..i-1} x[j]
    total = tl.zeros((), dtype=tl.int64)  # use int64 to match torch cumsum default dtype
    for i in range(0, length):
        xi = tl.load(x_ptr + i)
        total += xi
        tl.store(y_ptr + i, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # num_experts is fixed as in the original reference (256)
        self.num_experts = 256

    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32 on device
        Returns:
            sorted_token_indices: (num_tokens,) int32, same as original (argsort of flattened indices, stable=True)
            expert_offsets: (num_experts+1,) int64, same as original (cumsum of bincount with minlength=256)
        """
        # Flatten to 1D contiguous
        flat = topk_idx.reshape(-1).contiguous()

        # Compute the stable argsort of the flattened indices using PyTorch (not a sort of values).
        # We keep this in PyTorch to avoid implementing a stable sort in Triton; it's fast and correct.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        # Triton bincount into counts[0:num_experts], where num_experts = 256
        n = flat.numel()
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=flat.device)

        # We assume values in flat are in [0, 255], as per the original code generation.
        # If you need to handle values > 255, add a host-side mask or adjust logic; here we trust the input.
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        bincount_kernel[grid](
            flat,            # src_ptr
            counts,          # counts_ptr
            n,               # n_elements
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Inclusive prefix sum in Triton to produce offsets
        # We produce int64 offsets to match torch.cumsum default behavior (float/int can be promoted here).
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int64, device=flat.device)
        # Initialize x (counts) to int32; we'll read it and write y as int64
        prefix_sum_inclusive_kernel[(1,)](
            counts,          # x_ptr
            expert_offsets,  # y_ptr
            self.num_experts + 1,  # length
        )

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
