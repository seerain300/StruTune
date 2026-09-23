import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_sort_stable_asc_triton(indices_ptr, out_ptr, p_ptr, NUM_TOKS: tl.constexpr):
    """
    Triton kernel: stable odd-even transposition sort.
    - indices_ptr: pointer to int32 array of length NUM_TOKS (values to sort)
    - out_ptr: pointer to int32 array of length NUM_TOKS (sorted output)
    - p_ptr: pointer to int32 array of length NUM_TOKS (permutation, initially 0..NUM_TOKS-1)
    - NUM_TOKS: total number of elements (constexpr for Triton control flow).
    """
    # Perform NUM_TOKS passes; in each pass, alternate even and odd phases.
    for t in range(NUM_TOKS):
        if (t % 2) == 0:
            # Even phase: compare i and i+1 for even i
            for i in range(0, NUM_TOKS, 2):
                a = tl.load(indices_ptr + i)
                b = tl.load(indices_ptr + (i + 1))
                cond = a > b  # stable: no swap when equal
                minv = tl.where(cond, b, a)
                maxv = tl.where(cond, a, b)
                # write to output
                tl.store(out_ptr + i, minv)
                tl.store(out_ptr + (i + 1), maxv)
                # update permutation
                pi = tl.load(p_ptr + i)
                pi_next = tl.load(p_ptr + (i + 1))
                tl.store(p_ptr + i, tl.where(cond, pi_next, pi))
                tl.store(p_ptr + (i + 1), tl.where(cond, pi, pi_next))
            # Copy out back to indices_ptr for next phase
            for i in range(0, NUM_TOKS):
                val = tl.load(out_ptr + i)
                tl.store(indices_ptr + i, val)
        else:
            # Odd phase: compare i and i+1 for odd i
            for i in range(1, NUM_TOKS, 2):
                a = tl.load(indices_ptr + i)
                b = tl.load(indices_ptr + (i + 1))
                cond = a > b  # stable: no swap when equal
                minv = tl.where(cond, b, a)
                maxv = tl.where(cond, a, b)
                # write to output
                tl.store(out_ptr + i, minv)
                tl.store(out_ptr + (i + 1), maxv)
                # update permutation
                pi = tl.load(p_ptr + i)
                pi_next = tl.load(p_ptr + (i + 1))
                tl.store(p_ptr + i, tl.where(cond, pi_next, pi))
                tl.store(p_ptr + (i + 1), tl.where(cond, pi, pi_next))
            # Copy out back to indices_ptr
            for i in range(0, NUM_TOKS):
                val = tl.load(out_ptr + i)
                tl.store(indices_ptr + i, val)


@triton.jit
def _hist_kernel(indices_ptr, counts_ptr, NUM_TOKS: tl.constexpr, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton histogram kernel: counts per expert id.
    - indices_ptr: pointer to int32 flat array of length NUM_TOKS
    - counts_ptr: pointer to int32 array of length NUM_EXPERTS
    - NUM_TOKS: total number of elements
    - NUM_EXPERTS: number of experts (256 here)
    - BLOCK_SIZE: elements processed per program instance
    Each program processes BLOCK_SIZE elements, does local per-expert reductions, then atomically adds to counts.
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_TOKS
    vals = tl.load(indices_ptr + offs, mask=mask, other=0)  # int32
    # Compute per-expert partial counts and atomically add to global counts
    for i in range(NUM_EXPERTS):
        # count how many in this block equal i
        eq = vals == i
        # sum of booleans
        partial = tl.sum(eq.to(tl.int32), axis=0)
        # atomic add to global counts[i]
        tl.atomic_add(counts_ptr + i, partial)


@triton.jit
def _prefix_sum_inclusive_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    """
    Triton kernel computing inclusive prefix sum of counts into offsets_ptr.
    - counts_ptr: int32 array of length NUM_EXPERTS
    - offsets_ptr: int32 array of length NUM_EXPERTS + 1
    offsets[0] = 0, offsets[i+1] = sum_{j=0..i} counts[j]
    """
    # First write zeros to offsets
    for i in range(NUM_EXPERTS + 1):
        tl.store(offsets_ptr + i, 0)
    # Sequential inclusive scan
    running = 0
    for i in range(NUM_EXPERTS):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + (i + 1), running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-ONLY implementation:
        - Sorts flattened indices stably in Triton.
        - Computes per-expert counts via Triton histogram.
        - Computes inclusive prefix sums of counts via Triton kernel.
        """
        # Ensure on CUDA and contiguous
        assert topk_idx.is_cuda, "topk_idx must be on CUDA device for Triton kernels."
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # 1) Stable sort via Triton
        flat_copy = flat.to(torch.int32)
        out_flat = torch.empty(N, dtype=torch.int32, device=flat.device)
        p = torch.arange(N, dtype=torch.int32, device=flat.device)  # initial permutation 0..N-1
        _odd_even_sort_stable_asc_triton[(1,)](flat_copy, out_flat, p, NUM_TOKS=N)

        # sorted_token_indices is the permutation p (int32)
        sorted_token_indices = p

        # 2) Histogram per expert id using Triton
        num_experts = 256  # consistent with the original code's num_experts
        counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
        BLOCK_SIZE = 1024  # process 1024 elements per program instance
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _hist_kernel[grid](flat_copy, counts, NUM_TOKS=N, NUM_EXPERTS=num_experts, BLOCK_SIZE=BLOCK_SIZE)

        # 3) Inclusive prefix sums via Triton
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
        _prefix_sum_inclusive_kernel[(1,)](counts, expert_offsets, NUM_EXPERTS=num_experts)

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
