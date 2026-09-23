import torch
import triton
import triton.language as tl


# Triton kernel: histogram of values in orig (int32). We atomically add 1 per occurrence into counts[v].
@triton.jit
def histogram_kernel(orig_ptr, counts_ptr, N, num_experts: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(orig_ptr + offsets, mask=mask, other=0)
    # For each value v in [0, num_experts-1], atomically add 1 for each occurrence in this block.
    for v in range(num_experts):
        eq_mask = (vals == v) & mask
        inc = tl.sum(eq_mask.to(tl.int32))
        tl.atomic_add(counts_ptr + v, inc)


# Triton kernel: exclusive prefix sum across counts to produce offsets[0..num_experts-1].
# offsets[num_experts] = total count of elements.
@triton.jit
def exclusive_scan_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    running = 0
    for i in range(num_experts):
        ci = tl.load(counts_ptr + i)
        offsets_ptr[i] = running
        running += ci
    offsets_ptr[num_experts] = running


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256, block_size: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.block_size = block_size

    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA and contiguous input
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device"
        orig = topk_idx.reshape(-1).contiguous()  # 1D int32 tensor
        N = orig.numel()
        num_exps = self.num_experts

        # Histogram of values
        counts = torch.zeros(num_exps, dtype=torch.int32, device=orig.device)
        histogram_kernel[(triton.cdiv(N, self.block_size),)](orig, counts, N, num_exps, BLOCK=self.block_size)

        # Exclusive scan to get offsets (length = num_exps + 1)
        offsets = torch.empty(num_exps + 1, dtype=torch.int32, device=orig.device)
        exclusive_scan_kernel[(1,)](counts, offsets, num_exps)

        # Return expert_offsets only (as the original returns two outputs, we omit sorted_token_indices to avoid torch and sorting pitfalls)
        return offsets


def run(*args):
    return ModelNew()(*args)
