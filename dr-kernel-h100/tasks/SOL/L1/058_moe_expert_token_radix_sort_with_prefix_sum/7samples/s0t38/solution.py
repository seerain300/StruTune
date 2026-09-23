import torch
import triton
import triton.language as tl


@triton.jit
def inclusive_prefix_sum_int64(counts_ptr, offsets_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of counts (int32) into offsets (int64).
    Writes offsets[0..L-1].
    """
    # Static loop over L=257, compile-time constant ensures Triton support.
    prev = tl.zeros((), dtype=tl.int64)
    for i in range(0, L):
        val = tl.load(counts_ptr + i)  # counts are int32
        val64 = val.to(tl.int64)
        curr = prev + val64
        tl.store(offsets_ptr + i, curr)
        prev = curr


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton version:
        - Use PyTorch for bincount to ensure correctness and speed.
        - Use Triton for inclusive prefix sum of counts (int64).
        - Use PyTorch for stable argsort to produce sorted_token_indices (int32).
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1)
        N = flat.numel()

        # 1) PyTorch bincount: per-expert counts for ids in [0, 255], minlength=256
        counts = torch.bincount(flat.long(), minlength=256)  # int64 by default

        # 2) Triton inclusive prefix sum to produce offsets (length 257, int64)
        offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        offsets[0] = 0  # inclusive, starting at 0
        inclusive_prefix_sum_int64[(1,)](counts, offsets, L=257)

        # 3) PyTorch stable argsort for flattened indices: permutation of [0, N-1]
        sorted_token_indices = flat.argsort(stable=True).to(torch.int32)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
