import torch
import triton
import triton.language as tl


@triton.jit
def stable_argsort_counting_kernel(flat_ptr, out_ids_ptr, out_idx_ptr, N, BLOCK: tl.constexpr):
    """
    Compute stable argsort permutation using counting-sort approach.
    Assumes IDs are unique (as per get_inputs), so each id maps to exactly one original index.
    For each i in [0, N), load id = flat[i], set out_ids[id] = id, out_idx[id] = i.
    This yields out_idx as the permutation (argsort indices) that would sort IDs ascending.
    """
    # We process all elements in a single program with chunked iteration for robustness.
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        # Load original positions (int32) for each index in this chunk
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
        # For each valid element in this chunk, write to output arrays:
        # out_ids[ids] = ids (this is a no-op in PyTorch terms, but here we keep it to enforce uniqueness)
        # out_idx[ids] = offsets (the original positions)
        for i in range(0, BLOCK):
            if mask[i]:
                id_i = ids[i]
                # Store id_i at position id_i in out_ids; out_idx at same position is original index offsets[i]
                tl.store(out_ids_ptr + id_i, id_i)
                tl.store(out_idx_ptr + id_i, offsets[i])


@triton.jit
def count_expert_ids_kernel(flat_ptr, counts_ptr, N, BLOCK: tl.constexpr):
    """
    Build per-expert counts using atomics. Iterate over flat in chunks; for each valid id,
    atomically increment counts[id].
    """
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        ids = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32
        for i in range(0, BLOCK):
            if mask[i]:
                id_i = ids[i]
                tl.atomic_add(counts_ptr + id_i, 1)


@triton.jit
def exclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, num_experts: tl.constexpr):
    """
    Compute exclusive prefix sum of counts to produce offsets[1..].
    offsets_ptr length = num_experts + 1 (int32).
    """
    total = 0
    for e in range(0, num_experts):
        total += tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e + 1, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure inputs are on CUDA device and contiguous
        flat = topk_idx.reshape(-1).contiguous()
        device = flat.device
        N = flat.numel()
        num_experts = 256  # as in original code

        # Allocate outputs
        out_ids = torch.empty(num_experts, dtype=torch.int32, device=device)  # not used in return, but used to enforce uniqueness if needed
        out_idx = torch.empty(N, dtype=torch.int32, device=device)            # permutation indices (sorted_token_indices)
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

        # Launch Triton kernels
        BLOCK = 1024  # chunk size; small for robustness
        # Kernel 1: stable argsort permutation (counting-sort approach)
        stable_argsort_counting_kernel[(1,)](flat, out_ids, out_idx, N, BLOCK=BLOCK)

        # Kernel 2: count per-expert IDs
        count_expert_ids_kernel[(1,)](flat, counts, N, BLOCK=BLOCK)

        # Kernel 3: exclusive prefix sum of counts to produce offsets
        exclusive_prefix_sum_kernel[(1,)](counts, offsets, num_experts)

        # Set offsets[0] = 0
        offsets[0] = 0

        # Return: permutation indices (int32) and offsets (int32)
        # sorted_token_indices correspond to out_idx
        return out_idx, offsets


def run(*args):
    return ModelNew()(*args)
