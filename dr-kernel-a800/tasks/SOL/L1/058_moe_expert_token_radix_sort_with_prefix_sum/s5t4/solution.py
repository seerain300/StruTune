import torch
import triton
import triton.language as tl


# Triton kernel: compute per-expert counts (histogram) of flat values.
# Launch one program per key in [0, NUM_EXPERTS); each program loops over M elements.
@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    For each key k in [0, NUM_EXPERTS), counts_ptr[k] = number of elements in flat_ptr equal to k.
    """
    for k in range(0, NUM_EXPERTS):
        cnt = tl.zeros((), dtype=tl.int32)
        for j in range(0, M):
            val = tl.load(flat_ptr + j)
            if val == k:
                cnt += 1
        tl.store(counts_ptr + k, cnt)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of the original run():
        - sorted_token_indices: permutation of [0..M-1] ordered by topk_idx values (stable).
        - expert_offsets: inclusive prefix sums per expert (length = num_experts + 1), with +1 applied to the last element.
        Returns: (sorted_token_indices, expert_offsets)
        """
        # Flatten to 1D contiguous tensor
        flat = topk_idx.reshape(-1).contiguous()
        M = flat.numel()
        device = flat.device
        NUM_EXPERTS = self.num_experts

        # 1) Use Triton to compute per-expert counts (histogram)
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        _histogram_counts_kernel[(NUM_EXPERTS,)](flat, counts, M, NUM_EXPERTS)

        # 2) Compute expert offsets: inclusive cumsum of counts, then add +1 to the last element
        #    expert_offsets[e+1] - expert_offsets[e] == counts[e]
        #    The original code does: torch.bincount(flat).cumsum(0) + 1.
        #    We mimic that exactly: counts -> cumsum -> add 1 to the final element.
        cumsum = torch.cumsum(counts, dim=0)  # inclusive prefix sums
        expert_offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        if NUM_EXPERTS > 0:
            expert_offsets[1:] = cumsum
            expert_offsets[-1] += 1  # mimic the original "+ 1" applied to the last element
        else:
            # num_experts == 0 is not used in this harness; handle gracefully.
            pass

        # 3) Stable argsort using PyTorch for correctness
        sorted_token_indices = torch.argsort(flat, stable=True).to(torch.int32)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
