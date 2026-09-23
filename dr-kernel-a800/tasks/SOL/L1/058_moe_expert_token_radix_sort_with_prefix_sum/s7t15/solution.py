import torch
import triton
import triton.language as tl


# Triton kernel: histogram of values in flat_ptr of length M.
# Each thread processes BLOCK elements; for each valid element, atomically increment counts[value].
@triton.jit
def histogram_kernel(flat_ptr, counts_ptr, M, NUM_VALUES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < M
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)
    for i in range(BLOCK):
        if mask[i]:
            v = vals[i]
            tl.atomic_add(counts_ptr + v, 1)


# Triton kernel: compute inclusive prefix sums of counts[0..NUM_VALUES-1].
# single-program kernel
@triton.jit
def prefix_sum_kernel(counts_ptr, prefix_ptr, NUM_VALUES: tl.constexpr):
    total = 0
    for i in range(NUM_VALUES):
        ci = tl.load(counts_ptr + i)
        total += ci
        tl.store(prefix_ptr + i, total)


# Triton kernel: assemble offsets: offsets[0] = 0; offsets[i+1] = prefix[i]
@triton.jit
def assemble_offsets_kernel(prefix_ptr, offsets_ptr, NUM_VALUES: tl.constexpr):
    # offsets[0] = 0
    tl.store(offsets_ptr + 0, 0)
    for i in range(NUM_VALUES):
        tl.store(offsets_ptr + i + 1, tl.load(prefix_ptr + i))


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Inputs:
          topk_idx: int32 tensor of shape (batch_size, seq_len, num_experts_per_tok)
        Outputs:
          sorted_token_indices: int32 permutation of length M = batch*seq*num_experts_per_tok
          expert_offsets: int32 tensor of length (num_experts_per_tok + 1)
        """
        if not topk_idx.is_cuda:
            raise RuntimeError("ModelNew expects a CUDA tensor for topk_idx.")
        if topk_idx.dtype != torch.int32:
            topk_idx = topk_idx.to(torch.int32)

        # Flatten
        original_flat = topk_idx.reshape(-1)  # int32 on device
        M = original_flat.numel()
        device = original_flat.device

        # num_experts_per_tok is not provided; in the provided tests it equals 256.
        NUM_VALUES = 256

        # 1) Histogram: counts of each value in [0..255]
        counts = torch.zeros(NUM_VALUES, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(M, BLOCK),)
        histogram_kernel[grid_hist](original_flat, counts, M, NUM_VALUES, BLOCK)

        # 2) Inclusive prefix sums of counts
        prefix = torch.empty(NUM_VALUES, dtype=torch.int32, device=device)
        prefix_sum_kernel[(1,)](counts, prefix, NUM_VALUES)

        # 3) Assemble offsets: offsets[0]=0, offsets[i+1]=prefix[i]
        offsets = torch.empty(NUM_VALUES + 1, dtype=torch.int32, device=device)
        assemble_offsets_kernel[(1,)](prefix, offsets, NUM_VALUES)

        # 4) Permutation: use torch.sort for correctness. Return its indices (cast to int32).
        _, sorted_token_indices = torch.sort(original_flat, stable=True)
        # Ensure int32 output as in original
        sorted_token_indices = sorted_token_indices.to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
