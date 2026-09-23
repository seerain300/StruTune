import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, NUM_TOKS: tl.constexpr, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton histogram kernel:
    - flat_ptr: pointer to 1D int32 array of length NUM_TOKS
    - counts_ptr: pointer to 1D int32 array of length NUM_EXPERTS
    Each program instance processes BLOCK_SIZE elements, computes per-expert counts locally, and atomically adds to counts_ptr.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_TOKS

    # Load a block of values
    vals = tl.load(flat_ptr + offsets, mask=mask, other=0)  # int32

    # For each expert id, count how many times it appears in this block.
    # We loop over NUM_EXPERTS and use masked atomic_add to accumulate.
    for i in range(NUM_EXPERTS):
        # Build a boolean mask: vals == i
        is_i = vals == i
        # Sum masked values: 1 where equal, 0 otherwise
        # tl.sum reduces the BLOCK_SIZE vector to a scalar.
        count_i = tl.sum(is_i.to(tl.int32), axis=0)
        # Atomically add to global counts[i]
        tl.atomic_add(counts_ptr + i, count_i)


@triton.jit
def _prefix_sum_inclusive_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    """
    Triton inclusive prefix sum kernel over a small fixed-size array:
    - counts_ptr: pointer to 1D int32 array of length NUM_EXPERTS
    - offsets_ptr: pointer to 1D int32 array of length NUM_EXPERTS+1
    Compute offsets[j] = sum_{i=0..j} counts[i], with offsets[0] = 0.
    """
    # We process one element per loop; NUM_EXPERTS is constexpr so Triton unrolls efficiently.
    acc = 0
    for j in range(NUM_EXPERTS):
        acc += tl.load(counts_ptr + j)  # scalar load and add
        tl.store(offsets_ptr + j + 1, acc)  # store to j+1 index


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure CUDA and contiguous
        assert topk_idx.is_cuda, "Input must be on CUDA device"
        flat = topk_idx.reshape(-1).contiguous()
        # Flatten as int32 for uniformity
        flat = flat.to(torch.int32)

        N = flat.numel()
        num_experts = 256

        # 1) Stable sorting permutation via PyTorch (returns indices)
        # argsort gives indices that would sort the flattened values ascending.
        sorted_indices = torch.argsort(flat)

        # 2) Triton histogram to compute counts per expert
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_SIZE = 1024  # process 1024 elements per program instance
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _hist_kernel[grid](flat, counts, NUM_TOKS=N, NUM_EXPERTS=num_experts, BLOCK_SIZE=BLOCK_SIZE)

        # 3) Triton inclusive prefix sums to produce expert offsets
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        _prefix_sum_inclusive_kernel[(1,)](counts, expert_offsets, NUM_EXPERTS=num_experts)

        # sorted_token_indices should be the permutation; original code uses stable sort of 'flat',
        # which argsort emulates: return the indices.
        sorted_token_indices = sorted_indices

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
