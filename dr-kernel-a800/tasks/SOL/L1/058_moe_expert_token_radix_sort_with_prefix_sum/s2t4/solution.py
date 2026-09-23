import torch
import triton
import triton.language as tl


@triton.jit
def count_experts_kernel(vals_ptr, counts_ptr, N, num_experts: tl.constexpr):
    """
    Triton kernel to count occurrences of each expert id in vals_ptr (int32).
    For each loaded element v in [0, num_experts), atomic_add counts[v] += 1.
    vals_ptr points to flattened values in [0, num_experts).
    """
    pid = tl.program_id(0)
    BLOCK = 1024
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(vals_ptr + offsets, mask=mask, other=0)  # int32 values to be counted
    # Atomic add count for each expert id x
    tl.atomic_add(counts_ptr + x.to(tl.int32), 1, mask=mask)


@triton.jit
def prefix_sums_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Triton kernel to compute prefix sums of counts into offsets:
      offsets[0] = 0
      offsets[i+1] = offsets[i] + counts[i]  for i in 0..num_experts-1
    This is a simple sequential loop in Triton (num_experts is small).
    """
    # Start with offsets[0] = 0
    tl.store(offsets_ptr + 0, tl.zeros((), dtype=tl.int32))
    # Iterate and accumulate
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, num_experts):
        cnt = tl.load(counts_ptr + i)
        acc += cnt
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 256  # matches the original usage

    def forward(self, topk_idx: torch.Tensor):
        # Ensure input is on CUDA and contiguous
        assert topk_idx.is_cuda, "ModelNew.forward expects a CUDA tensor"
        device = topk_idx.device

        # Flatten the indices
        vals = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = vals.numel()

        # 1) Count per-expert occurrences using Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        count_experts_kernel[grid](vals, counts, N, self.num_experts)

        # 2) Compute expert offsets (prefix sums) using Triton kernel
        expert_offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        prefix_sums_kernel[(1,)](counts, expert_offsets, self.num_experts)

        # 3) sorted_token_indices: use torch.sort (stable=True) to match original behavior exactly
        #    This is necessary for correctness; Triton does not provide a built-in sort.
        sorted_token_indices = torch.sort(vals, stable=True).indices

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
