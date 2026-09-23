import torch
import triton
import triton.language as tl


# Triton kernel: histogram of values in original_flat (int32).
# Each program processes BLOCK elements, atomically increments counts[value].
@triton.jit
def histogram_kernel(original_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(original_ptr + offsets, mask=mask, other=0)  # int32
    for i in range(BLOCK):
        if mask[i]:
            v = vals[i]
            # Triton int32 atomic add
            tl.atomic_add(counts_ptr + v, 1)


# Triton kernel: compute inclusive prefix sum of counts (length NUM_VALUES).
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    # Single program computes the prefix sum sequentially.
    for i in range(NUM_VALUES):
        # Load previous sum
        prev = prefix_ptr[i - 1] if i > 0 else 0
        ci = tl.load(counts_ptr + i)
        prefix_ptr[i] = prev + ci


# Triton kernel: assemble offsets from prefix (length NUM_VALUES+1).
@triton.jit
def assemble_offsets_kernel(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    # offsets[0] = 0
    # offsets[i+1] = prefix[i] for i in [0..NUM_VALUES-1]
    offsets_ptr[0] = 0
    for i in range(NUM_VALUES):
        offsets_ptr[i + 1] = tl.load(prefix_ptr + i)


# Triton kernel: stable permutation for values in [0..NUM_VALUES-1].
# For each value v, we place indices of v at positions:
#   number_of_less + number_of_equal_before_i (stable tie-break by original position).
@triton.jit
def stable_permute_kernel(original_ptr, sorted_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    # We iterate v from 0..NUM_VALUES-1
    for v in range(NUM_VALUES):
        # Compute number_of_less (elements strictly less than v)
        number_of_less = 0
        base = 0
        while base < M:
            offsets = base + tl.arange(0, BLOCK)
            mask = offsets < M
            vals = tl.load(original_ptr + offsets, mask=mask, other=0)  # int32
            # count of vals < v
            less = 0
            for i in range(BLOCK):
                if mask[i]:
                    if vals[i] < v:
                        less += 1
            base += BLOCK
        # Now place indices: for each i, if original[i] == v, put i at position
        # number_of_less + number_of_equal_before_i (stable by original position).
        base = 0
        while base < M:
            offsets = base + tl.arange(0, BLOCK)
            mask = offsets < M
            vals = tl.load(original_ptr + offsets, mask=mask, other=0)  # int32
            for i in range(BLOCK):
                if mask[i]:
                    vi = vals[i]
                    pos_i = offsets[i]
                    if vi == v:
                        # number_of_equal_before_i: count equal values before pos_i
                        eq_before = 0
                        # scan original_ptr[0:pos_i) to count equals
                        j = 0
                        while j < pos_i:
                            vj = tl.load(original_ptr + j, mask=j<pos_i, other=0)  # scalar load
                            if vj == v:
                                eq_before += 1
                            j += 1
                        out_pos = number_of_less + eq_before
                        tl.store(sorted_ptr + out_pos, pos_i)
            base += BLOCK


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: int32 tensor of shape (batch_size, seq_len, num_experts_per_tok)
        Returns:
        - sorted_token_indices: int32 tensor of shape (M,) = argsort permutation of flattened topk_idx stable by value.
        - expert_offsets: int32 tensor of shape (num_experts_per_tok + 1,) = inclusive count per value [0..num_experts_per_tok-1]
        """
        device = topk_idx.device
        original_flat = topk_idx.reshape(-1).contiguous()  # int32, 1D
        M = original_flat.numel()
        NUM_VALUES = 256  # consistent with get_inputs and original code

        # Allocate outputs
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)

        # 1) Histogram via Triton
        BLOCK = 1024
        grid_hist = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid_hist](original_flat, counts, M, NUM_VALUES, BLOCK)

        # 2) Inclusive prefix sum via Triton
        prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES)

        # 3) Assemble offsets
        assemble_offsets_kernel[(1,)](prefix, offsets, NUM_VALUES)

        # 4) Stable permutation via Triton
        BLOCK_PERM = 1024
        grid_perm = (triton.cdiv(M, BLOCK_PERM),)
        stable_permute_kernel[grid_perm](original_flat, sorted_token_indices, M, NUM_VALUES, BLOCK_PERM)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
