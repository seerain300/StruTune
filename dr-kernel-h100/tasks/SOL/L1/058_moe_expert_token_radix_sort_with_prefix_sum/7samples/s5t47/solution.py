import torch
import triton
import triton.language as tl


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    # Compute exclusive prefix sum for offsets[1..] given counts[0..num_experts-1].
    # offsets[i+1] = sum(counts[0..i]) for i in [0..num_experts-1].
    # offsets[0] is set to 0 on the host.
    sum_val = tl.zeros((), dtype=tl.int32)  # scalar int32 accumulator
    for i in range(num_experts):
        # Load current count
        cnt = tl.load(counts_ptr + i)
        # Store exclusive prefix sum at i+1
        tl.store(offsets_ptr + (i + 1), sum_val)
        # Accumulate for next iteration
        sum_val += cnt


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D int32 on device
        flat = topk_idx.reshape(-1).to(torch.int32)

        # Compute histogram with torch (fast and robust)
        num_experts = 256
        counts = torch.bincount(flat.long(), minlength=num_experts)  # int64 by default, fine for sum

        # Prepare offsets on device
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

        # Launch Triton kernel for exclusive prefix sum into offsets[1..]
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)
        # Set offsets[0] = 0
        offsets[0] = 0

        # Compute sorted token indices using torch (stable) on the flattened tensor
        sorted_token_indices = flat.argsort(stable=True)

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets


def run(*args):
    return ModelNew()(*args)
