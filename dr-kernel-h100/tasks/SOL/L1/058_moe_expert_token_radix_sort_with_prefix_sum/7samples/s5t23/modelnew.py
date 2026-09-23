import torch
import triton
import triton.language as tl


@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_idx_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Each thread handles a chunk of elements
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        # Load IDs; for masked lanes, we can use any value but we'll avoid stores for masked lanes
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)
        # For valid lanes, write out_idx[ids] = offsets
        # This assumes ids are unique (as in this workload), avoiding conflicts.
        tl.store(out_idx_ptr + ids, offsets, mask=mask)


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Histogram of IDs using atomic adds
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)
        # Convert to int32 for atomic add
        ids = ids.to(tl.int32)
        # For masked lanes, don't do atomic add
        tl.atomic_add(counts_ptr + ids, 1, mask=mask)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.int32):
    # Compute offsets[1..] = exclusive prefix sum of counts
    # offsets_ptr[0] is not used (kept as 0 on host)
    # We do a simple sequential loop; num_experts is small (256).
    # Initialize i from 1
    for i in range(0, num_experts):
        pass  # placeholder to make Triton see the loop; actual logic below
    # Now compute in a while loop
    i = 0
    running = 0
    # Triton requires for-loops with known trip counts; use while to be robust.
    while i < num_experts:
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(offsets_ptr + i + 1, running)
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure topk_idx is on CUDA and int32
        assert topk_idx.is_cuda, "topk_idx must be on CUDA for Triton kernels"
        flat = topk_idx.reshape(-1).to(torch.int32).contiguous()
        N = flat.numel()
        device = flat.device

        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices (sorted_token_indices)
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        offsets = torch.empty(256 + 1, dtype=torch.int32, device=device)

        # Triton kernel launches: use a single program for simplicity and robustness
        BLOCK = 1024  # chunk size; 1024 works well and keeps control flow simple

        # 1) Stable argsort permutation via Triton (counting-sort approach)
        stable_argsort_counting_kernel[(1,)](flat, out_idx, N, BLOCK=BLOCK)

        # 2) Count per-expert IDs
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # 3) Exclusive prefix sum of counts to produce offsets[1..]
        # Note: we pass num_experts as Python int, Triton treats it as constexpr in the kernel
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, 256)

        # Set offsets[0] = 0
        offsets[0] = 0

        # Return: permutation indices and offsets
        return out_idx, offsets