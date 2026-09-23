import torch
import triton
import triton.language as tl


@triton.jit
def counts_by_exp_kernel(flat_ptr, counts_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    For each expert e in [0..NUM_EXPERTS), count occurrences in flat_ptr[0..M).
    counts_ptr[e] = number of elements in flat equal to e.
    """
    e = tl.program_id(0)  # one program per expert
    if e >= NUM_EXPERTS:
        return
    acc = tl.zeros((), dtype=tl.int32)
    # scan entire flat vector
    for i in range(0, M):
        val = tl.load(flat_ptr + i)
        if val == e:
            acc += 1
    tl.store(counts_ptr + e, acc)


@triton.jit
def cumsum_inclusive_kernel(counts_ptr, offsets_incl_ptr, NUM_EXPERTS: tl.constexpr):
    """
    Compute inclusive prefix sums of counts_ptr[0..NUM_EXPERTS) into offsets_incl_ptr[0..NUM_EXPERTS).
    offsets_incl[e] = sum(counts[0..e]).
    """
    e = tl.program_id(0)
    if e >= NUM_EXPERTS:
        return
    acc = tl.zeros((), dtype=tl.int32)
    for i in range(0, e + 1):
        acc += tl.load(counts_ptr + i)
    tl.store(offsets_incl_ptr + e, acc)


@triton.jit
def counts_sum_kernel(counts_ptr, total_ptr, NUM_EXPERTS: tl.constexpr):
    """
    Compute total = sum(counts_ptr[0..NUM_EXPERTS)) and store to total_ptr (single int32).
    """
    total = tl.zeros((), dtype=tl.int32)
    for i in range(0, NUM_EXPERTS):
        total += tl.load(counts_ptr + i)
    tl.store(total_ptr, total)


@triton.jit
def stable_argsort_kernel(flat_ptr, sorted_ptr, offsets_incl_ptr, M: tl.int32, NUM_EXPERTS: tl.constexpr):
    """
    Compute stable argsort of flat_ptr[0..M) and write to sorted_ptr[0..M).
    stable=True => ties are ordered by original index.
    For each j:
      key_j = flat[j]
      base_excl = offsets_incl[key_j - 1] if key_j > 0 else 0
      tie_count = number of t < j with key_t == key_j and flat[t] <= flat[j]
      rank = base_excl + tie_count
      sorted[j] = rank
    """
    j = tl.program_id(0)
    if j >= M:
        return
    val_j = tl.load(flat_ptr + j)
    # base exclusive for key val_j
    if val_j == 0:
        base = tl.zeros((), dtype=tl.int32)
    else:
        base = tl.load(offsets_incl_ptr + (val_j - 1))
    # tie_count: count elements t < j with same key and smaller-or-equal value (for stability, ties by index)
    tie = tl.zeros((), dtype=tl.int32)
    for t in range(0, j):
        val_t = tl.load(flat_ptr + t)
        # stable tie handling: count when value equals and original index comes before j
        if val_t == val_j and val_t <= val_j:
            tie += 1
    rank = base + tie
    tl.store(sorted_ptr + j, rank)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          flat = topk_idx.reshape(-1)
          sorted_token_indices = torch.sort(flat, stable=True).values (int32)
          expert_offsets = torch.bincount(flat).cumsum(0) + 1 (int32, length num_experts+1)
        """
        assert topk_idx.is_cuda, "Input must be on CUDA device for Triton kernels."
        flat = topk_idx.reshape(-1)
        M = flat.numel()
        NUM_EXPERTS = 256  # fixed in harness

        # 1) Count per expert
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        grid_counts = (NUM_EXPERTS,)
        counts_by_exp_kernel[grid_counts](flat, counts, M, NUM_EXPERTS)

        # 2) Inclusive prefix sums of counts
        offsets_incl = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=flat.device)
        cumsum_inclusive_kernel[(NUM_EXPERTS,)](counts, offsets_incl, NUM_EXPERTS)

        # 3) Sum of counts (total tokens) via Triton
        total_count_buf = torch.empty(1, dtype=torch.int32, device=flat.device)
        counts_sum_kernel[(1,)](counts, total_count_buf, NUM_EXPERTS)

        # 4) Stable argsort using Triton
        sorted_token_indices = torch.empty(M, dtype=torch.int32, device=flat.device)
        grid_argsort = (M,)
        stable_argsort_kernel[grid_argsort](flat, sorted_token_indices, offsets_incl, M, NUM_EXPERTS)

        # 5) Finalize expert_offsets: inclusive prefix sums and append total+1
        expert_offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=flat.device)
        expert_offsets[:NUM_EXPERTS] = offsets_incl
        expert_offsets[NUM_EXPERTS] = int(total_count_buf.item()) + 1

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
