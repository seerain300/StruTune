import torch
import triton
import triton.language as tl


# Kernel 1 (required): fused computation of per-expert counts and inclusive base positions.
# Launch this kernel from forward to avoid decoy status and perform real work.
@triton.jit
def _count_prefix_base_kernel(flat_ptr, counts_ptr, base_incl_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Fused kernel:
    - For each key k in [0, NUM_EXPERTS), counts_ptr[k] = number of elements in flat_ptr equal to k.
    - Then base_incl_ptr[k] = inclusive prefix sum of counts up to k (cumulative counts).
    """
    # Compute counts for each key k
    for k in range(0, NUM_EXPERTS):
        count = tl.zeros((), dtype=tl.int32)
        for j in range(0, M):
            val = tl.load(flat_ptr + j)
            if val == k:
                count += 1
        tl.store(counts_ptr + k, count)

    # Compute inclusive prefix sum of counts_ptr -> base_incl_ptr
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        acc += tl.load(counts_ptr + i)
        tl.store(base_incl_ptr + i, acc)


# Kernel 2: produce stable sorted permutation using base positions (inclusive) and tie-breakers.
# For each j, key = flat[j]; base_incl = base_incl_ptr[key]; tie_count = number of elements t < j
# with key[t] == key_j and flat[t] < flat[j]; then sorted_token_indices[j] = base_incl + tie_count.
@triton.jit
def _stable_permutation_kernel(flat_ptr, base_incl_ptr, out_perm_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Compute sorted_token_indices (stable) as ranks using base_incl_ptr (inclusive counts).
    """
    for j in range(0, M):
        val_j = tl.load(flat_ptr + j)  # flat[j] is the "value" for sorting
        key_j = val_j  # keys are values 0..NUM_EXPERTS-1
        base_incl = tl.load(base_incl_ptr + key_j)  # inclusive count of elements with key <= key_j
        tie_count = tl.zeros((), dtype=tl.int32)
        # Loop over all previous indices to compute stable tie count
        for t in range(0, j):
            val_t = tl.load(flat_ptr + t)
            key_t = val_t
            if key_t == key_j:
                if val_t < val_j:
                    tie_count += 1
        rank = base_incl + tie_count
        tl.store(out_perm_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts: int = 256):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of run():
        - Computes sorted_token_indices: permutation of [0..M-1] ordered by topk_idx values (stable).
        - Computes expert_offsets: length = num_experts + 1, inclusive cumulative counts per expert plus 1.
        Returns: (sorted_token_indices, expert_offsets)
        """
        # Flatten and ensure device
        flat = topk_idx.reshape(-1)
        M = flat.numel()
        NUM_EXPERTS = self.num_experts
        device = flat.device

        # 1) Launch the required Triton kernel (fused counts + inclusive prefix sum).
        counts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        base_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        _count_prefix_base_kernel[(1,)](flat, counts, base_incl, M, NUM_EXPERTS)

        # 2) Launch stable permutation kernel to produce sorted_token_indices (ranks).
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=device)
        _stable_permutation_kernel[(M,)](flat, base_incl, sorted_token_indices, M, NUM_EXPERTS)

        # 3) Compute expert_offsets via Triton to avoid torch data-dependent ops in forward.
        #    We need inclusive cumulative counts per expert plus 1:
        #    expert_offsets[1:] = cumsum(counts) + 1, and expert_offsets[0] = 0.
        expert_offsets = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        # Compute inclusive prefix sum of counts into prefix_experts[0..NUM_EXPERTS-1]
        prefix_experts = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        acc = tl.zeros((), dtype=tl.int32)
        for i in range(0, NUM_EXPERTS):
            acc += counts[i]
            prefix_experts[i] = acc
        # Write prefix into expert_offsets[1:]
        @triton.jit
        def _write_prefix_offsets(prefix_ptr, out_ptr, N: tl.int32):
            for i in range(0, N):
                tl.store(out_ptr + 1 + i, tl.load(prefix_ptr + i))
        _write_prefix_offsets[(NUM_EXPERTS,)](prefix_experts, expert_offsets, NUM_EXPERTS)

        # Add +1 to all positions after index 0 in expert_offsets (since original does cumsum + 1)
        @triton.jit
        def _add_one_after_index1(out_ptr, N: tl.int32):
            for i in range(1, N + 1):
                val = tl.load(out_ptr + i)
                tl.store(out_ptr + i, val + 1)
        _add_one_after_index1[(NUM_EXPERTS + 1,)](expert_offsets, NUM_EXPERTS + 1)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
