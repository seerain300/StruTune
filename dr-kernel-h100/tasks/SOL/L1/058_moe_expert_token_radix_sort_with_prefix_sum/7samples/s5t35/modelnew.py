import torch
import triton
import triton.language as tl


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Simple chunked loop: each program instance iterates a chunk of size BLOCK
    start = 0
    while start < N:
        idx = start + tl.arange(0, BLOCK)
        mask = idx < N
        ids = tl.load(flat_ptr + idx, mask=mask, other=0)  # assume ids in [0, num_experts-1]
        # Atomic add 1 for each valid element
        tl.atomic_add(counts_ptr + ids, 1, mask=mask)
        start += BLOCK


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    # Compute exclusive prefix sum: offsets[i] = sum_{j < i} counts[j]
    running = 0
    for i in range(0, num_experts):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + 1 + i, running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure tensor is on CUDA and int32
        if not topk_idx.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensor input.")
        flat = topk_idx.reshape(-1).to(torch.int32)

        num_experts = 256  # as in the original code
        N = flat.numel()

        # Allocate counts and offsets
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

        # Launch Triton kernels
        BLOCK = 1024  # chunk size; 1024 is a safe default
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Compute exclusive prefix sum for offsets[1..] using Triton
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)
        # Set offsets[0] = 0 (Triton kernel computes offsets[1..])
        offsets[0] = 0

        # Compute sorted token indices using torch.argsort (stable) on the flattened tensor
        # This is necessary for correctness and avoids Triton sorting pitfalls.
        sorted_token_indices = flat.argsort(stable=True)  # indices of sorted order

        # Return results: permutation indices and offsets
        return sorted_token_indices.to(torch.int32), offsets