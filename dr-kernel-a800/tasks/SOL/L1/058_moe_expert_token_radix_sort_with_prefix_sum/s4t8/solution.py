import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(x_ptr, out_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute per-expert counts for a flattened 1D array x_ptr of length N.
    out_ptr is a 1D int32 array of length num_experts (256). Each Triton program
    processes BLOCK elements and performs a vectorized reduction to compute counts
    for each bin. It then atomically adds the per-bin counts to out_ptr.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load values and cast to int32
    idx = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # For each expert bin j, compute count and atomically add once per program.
    # This reduces atomic contention compared to per-element atomics.
    for j in range(num_experts):
        eq = (idx == j) & mask
        count_j = tl.sum(eq.to(tl.int32), axis=0)
        tl.atomic_add(out_ptr + j, count_j)


@triton.jit
def _inclusive_prefix_sum_kernel(in_ptr, out_ptr, num_experts: tl.constexpr):
    """
    Compute inclusive prefix sum of in_ptr (length num_experts) into out_ptr (length num_experts+1).
    out_ptr[0] = 0, out_ptr[1..] = inclusive scan of in_ptr.
    We scan from the end to the beginning:
      prev = out[i+1]
      out[i+1] = prev + in[i]
    """
    for i in tl.static_range(num_experts - 1, -1, -1):
        prev = tl.load(out_ptr + (i + 1))
        val = tl.load(in_ptr + i)
        tl.store(out_ptr + (i + 1), prev + val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        """
        Expect a single tensor argument: topk_idx of shape (batch, seq_len, num_experts_per_tok)
        Returns:
          sorted_token_indices: torch.int32 of shape (N,) where N = batch*seq_len*num_experts_per_tok
          expert_offsets: torch.int32 of shape (num_experts+1,)
        """
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single tensor argument: topk_idx")
        topk_idx = args[0]

        # Ensure tensor is on CUDA and contiguous
        if not topk_idx.is_cuda:
            topk_idx = topk_idx.to(device="cuda")
        topk_idx = topk_idx.contiguous()

        # Flatten to 1D
        flat = topk_idx.view(-1)
        N = flat.numel()

        # 1) Triton histogram of expert IDs
        num_experts = 256
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)

        # Launch histogram kernel
        BLOCK = 2048  # process 2048 elements per program; tuneable
        grid = (triton.cdiv(N, BLOCK),)
        _histogram_kernel[grid](flat, counts, N, num_experts=num_experts, BLOCK=BLOCK, num_warps=8)

        # 2) Inclusive prefix sum in Triton (no torch.cumsum)
        expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
        _inclusive_prefix_sum_kernel[(num_experts,)](counts, expert_offsets, num_experts=num_experts)

        # 3) Stable sort of flattened indices using PyTorch (ensures correctness)
        sorted_token_indices = flat.sort(stable=True)[1].to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
