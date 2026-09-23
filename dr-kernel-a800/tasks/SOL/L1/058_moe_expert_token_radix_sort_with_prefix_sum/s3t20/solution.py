import torch
import triton
import triton.language as tl


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, N_bins: tl.constexpr):
    """
    Compute exclusive prefix sum of counts_ptr[0:N_bins] and store in offsets_ptr[0:N_bins].
    offsets_ptr[0] = 0; offsets[i] = sum_{k=0..i-1} counts[k] for i>0.
    """
    # Host code sets offsets[0] = 0 before launching this kernel.
    for i in range(1, N_bins + 1):
        total = tl.zeros((), dtype=tl.int32)
        for k in range(0, i):
            total += tl.load(counts_ptr + k)
        tl.store(offsets_ptr + i, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, topk_idx: torch.Tensor):
        """
        Input: topk_idx of shape (batch_size, seq_len, num_experts_per_tok), int32.
        Output: (sorted_token_indices, expert_offsets)
        - sorted_token_indices: torch.long, shape (num_tokens,)
        - expert_offsets: torch.int32, shape (num_experts + 1,)
        """
        # Ensure input is on CUDA
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels"
        flat = topk_idx.reshape(-1).contiguous()
        num_tokens = flat.numel()
        num_experts = 256

        # 1) sorted_token_indices: use PyTorch's stable sort for correctness
        _, sorted_token_indices = flat.sort(stable=True)

        # 2) Compute counts via PyTorch bincount
        counts = torch.bincount(flat.long(), minlength=num_experts)

        # 3) Compute expert offsets using Triton (exclusive prefix sum)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        offsets[0] = 0  # exclusive prefix: start with 0
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, N_bins=num_experts, num_warps=1)

        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
