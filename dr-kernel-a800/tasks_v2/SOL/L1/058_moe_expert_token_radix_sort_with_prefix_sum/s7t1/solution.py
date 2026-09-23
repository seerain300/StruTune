import torch
import triton
import triton.language as tl


# Triton kernel: compute histogram of flattened indices (int32).
# For each element in flat, atomically increment counts[value].
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        v = vals[i]
        if mask[i]:
            tl.atomic_add(counts_ptr + v, 1)


# Triton kernel: compute inclusive prefix sum of counts -> prefix[v] = sum_{x<=v} counts[x]
# Writes to prefix_ptr[v] for v in [0..NUM_VALUES-1].
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    prefix = tl.zeros((), dtype=tl.int32)
    # Initialize prefix[0] = 0
    tl.atomic_add(prefix_ptr + 0, 0)
    i = 0
    while i < NUM_VALUES:
        count_i_minus_1 = tl.load(counts_ptr + i)  # counts[i] is number of 'i's
        tl.atomic_add(prefix_ptr + (i + 1), prefix)
        prefix += count_i_minus_1
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version:
        - sorted_token_indices: produced via torch.sort(stable=True) to ensure exact permutation.
        - expert_offsets: produced by Triton histogram + prefix sum, then assembled as in original.
        """
        if not topk_idx.is_cuda:
            raise RuntimeError("ModelNew expects topk_idx on a CUDA device.")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.view(-1)  # int32
        M = flat.numel()

        # num_experts_per_tok is the third dimension in the original input
        num_experts_per_tok = topk_idx.shape[2]

        # 1) sorted_token_indices: use PyTorch for exact stable permutation
        sorted_token_indices = torch.sort(flat.long(), stable=True).indices  # length M, int64 by default
        sorted_token_indices = sorted_token_indices.to(torch.int32)  # match original dtype

        # 2) Compute expert offsets using Triton histogram + prefix sum
        counts = torch.zeros(num_experts_per_tok, dtype=torch.int32, device=flat.device)

        # Triton histogram: one atomic add per element
        BLOCK = 1024
        grid = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid](flat, counts, M, NUM_VALUES=num_experts_per_tok, BLOCK=BLOCK)

        # Triton prefix sum of counts: inclusive prefix
        prefix = torch.empty(num_experts_per_tok, dtype=torch.int32, device=flat.device)
        prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES=num_experts_per_tok)

        # Assemble expert_offsets: original does expert_offsets[0] = 0; expert_offsets[1:] = prefix
        expert_offsets = torch.empty(num_experts_per_tok + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = prefix

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
