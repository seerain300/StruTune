import torch
import triton
import triton.language as tl


@triton.jit
def per_element_bincount_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    For each element in the flattened array, atomically increment the corresponding bin.
    Assumes flat_ptr values are in [0, 255] and counts_ptr has length 256 (int32).
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load values; ensure out-of-bounds elements don't affect counts
    vals = tl.load(flat_ptr + offs, mask=mask, other=0)
    # Atomically add 1 to counts for each valid value
    tl.atomic_add(counts_ptr + vals, 1, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) Triton per-element bincount into counts[0:256] (int32)
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        BLOCK = 1024  # reasonable chunk size for parallelism
        grid = (triton.cdiv(N, BLOCK),)
        per_element_bincount_kernel[grid](flat, counts, N, BLOCK=BLOCK)

        # 2) PyTorch bincount + cumsum to produce offsets (int64 of length 257)
        #    This matches torch.bincount(flat.long(), minlength=256).cumsum(0) exactly.
        counts_long = counts.to(torch.int64)
        expert_offsets = torch.bincount(counts_long, minlength=256).cumsum(0)  # int64

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        #    Original returns int32 for sorted_token_indices.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets