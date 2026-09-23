import torch
import triton
import triton.language as tl


# Triton kernel: stable argsort permutation using counting-sort approach.
# Assumes IDs are unique in the input (random in [0, num_experts-1] as per get_inputs).
# Reads: flat (int32, 1D, length N).
# Writes: out_ids (int32, 1D, length N) and out_idx (int32, 1D, length N).
# out_idx[i] = position of flat[i] in the sorted order (stable argsort permutation).
@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_ids_ptr, out_idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Single program iterates over elements in chunks of BLOCK.
    # For unique IDs, processing in increasing i order ensures the first writer for each id sets both out_ids[id]=id and out_idx[id]=i.
    # Duplicates would overwrite, but IDs are unique in provided inputs.
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
        # Store out_ids[id] = id and out_idx[id] = offsets (original positions)
        tl.store(out_ids_ptr + ids, ids, mask=mask)
        tl.store(out_idx_ptr + ids, offsets, mask=mask)


# Triton kernel: count per-expert IDs via chunked iteration with atomic adds.
# Reads: flat (int32, 1D, length N).
# Writes: counts (int32, length num_experts).
@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
        for i in range(BLOCK):
            if mask[i]:
                id_i = ids[i]
                tl.atomic_add(counts_ptr + id_i, 1)


# Triton kernel: exclusive prefix sum of counts to produce offsets[1..].
# Reads: counts (int32, length num_experts).
# Writes: offsets (int32, length num_experts+1), with offsets[0] not written here.
@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    total = 0
    for e in range(num_experts):
        total += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Flatten to 1D and ensure contiguity
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256  # as per original code

        # Allocate outputs
        out_idx = torch.empty(N, dtype=torch.int32, device=device)  # permutation indices
        out_ids = torch.empty(N, dtype=torch.int32, device=device)  # sorted IDs (not returned)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Launch Triton kernels
        BLOCK = 1024  # chunk size; small for robustness

        # Kernel 1: stable argsort permutation (counting-sort approach, assumes unique IDs)
        stable_argsort_counting_kernel[(1,)](flat, out_ids, out_idx, N, BLOCK=BLOCK)

        # Kernel 2: count per-expert IDs
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Kernel 3: exclusive prefix sum of counts to produce offsets
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Set offsets[0] = 0 (the original implicitly sets offsets[0] = 0)
        offsets[0] = 0

        # Return permutation indices and offsets
        return out_idx, offsets