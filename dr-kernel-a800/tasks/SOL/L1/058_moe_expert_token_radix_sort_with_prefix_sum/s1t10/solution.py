import triton
import triton.language as tl

# Fixed number of experts as in the original code
NUM_EXPERTS = 256


@triton.jit
def count_experts_kernel(flat_ptr, counts_ptr, N: tl.int32, BLOCK_COUNT: tl.constexpr):
    """
    Compute counts of each expert id in flat_ptr[0..N-1] into counts_ptr[0..NUM_EXPERTS-1].
    No atomics used. One program iterates over tiles of size BLOCK_COUNT and increments counts_ptr[id] for each element.
    """
    i = 0
    while i < N:
        # Process BLOCK_COUNT elements at a time
        for j in range(BLOCK_COUNT):
            idx = i + j
            m = idx < N
            if m:
                idv = tl.load(flat_ptr + idx)  # scalar int32 id
                # Increment counts_ptr[idv] by 1. Triton supports scalar pointer arithmetic.
                curr = tl.load(counts_ptr + idv)
                tl.store(counts_ptr + idv, curr + 1)
        i += BLOCK_COUNT


@triton.jit
def prefix_sum_kernel(counts_ptr, out_ptr, M: tl.int32):
    """
    Inclusive prefix sum of counts_ptr[0..M-1] into out_ptr[0..M-1].
    Single-program sequential scan. M is the number of experts (NUM_EXPERTS).
    """
    carry = tl.zeros((), dtype=tl.int32)
    for i in range(M):
        val = tl.load(counts_ptr + i)
        carry += val
        tl.store(out_ptr + i, carry)


@triton.jit
def compute_out_pos_real(flat_ptr, le_counts_ptr, lt_counts_ptr, out_ptr, N: tl.int32, BLOCK_OUT: tl.constexpr):
    """
    Compute stable argsort permutation: out[i] = position of flat[i] in stable order.
    Uses le_counts_ptr and lt_counts_ptr which have length NUM_EXPERTS.
    N: number of elements in flat_ptr/out_ptr
    BLOCK_OUT: tile size for processing indices
    """
    i = 0
    while i < N:
        for j in range(BLOCK_OUT):
            idx = i + j
            m = idx < N
            if m:
                idv = tl.load(flat_ptr + idx)  # scalar id
                lev = tl.load(le_counts_ptr + idv)  # scalar int32
                ltv = tl.load(lt_counts_ptr + idv)  # scalar int32
                duplicates = 1 if (ltv > 0) else 0  # scalar 0/1
                pos = lev - duplicates
                tl.store(out_ptr + idx, pos)
        i += BLOCK_OUT


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation:
        - Flatten topk_idx -> 1D int32 tensor of length N
        - Compute counts per expert (NUM_EXPERTS=256) using Triton
        - Compute le_counts and lt_counts via Triton inclusive scan
        - Compute stable argsort permutation via Triton
        - Produce expert_offsets as inclusive prefix sums of counts (without torch ops)
        Returns (sorted_token_indices, expert_offsets)
        """
        # Ensure input is contiguous and on device
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Compute counts per expert without atomics: one program iterates in tiles
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
        # Use a moderate tile


def run(*args):
    return ModelNew()(*args)
