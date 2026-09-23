import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_atomic_per_element_kernel(flat_ptr, counts_ptr, N):
    # Single program instance processes the input sequentially, issuing atomic_add per element.
    i = 0
    while i < N:
        val = tl.load(flat_ptr + i)
        tl.atomic_add(counts_ptr + val, 1)
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)
        N = flat.numel()
        device = flat.device

        # Triton buffer for counts (per expert)
        num_experts = 256  # matches the original setup
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)

        # Kernel: histogram via per-element atomic add
        count_expert_ids_atomic_per_element_kernel[(1,)](flat, counts, N)

        # Compute offsets via torch.cumsum (exclusive prefix): offsets[i] = sum_{j < i} counts[j]
        cumsum = torch.cumsum(counts, dim=0)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        offsets[1:] = cumsum  # copy inclusive prefix to offsets[1..]

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
