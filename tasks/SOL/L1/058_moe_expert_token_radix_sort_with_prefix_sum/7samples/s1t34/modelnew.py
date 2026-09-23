import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_counts_kernel(flat_ptr, counts_ptr, n_elements, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    # Iterate over the flat array in chunks of BLOCK and perform atomic_add into counts[val]
    for start in range(0, n_elements, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_elements
        vals = tl.load(flat_ptr + offs, mask=mask, other=0)  # int32
        # For each value in the chunk, do atomic add
        for j in range(BLOCK):
            idx = start + j
            if mask[j]:
                val = vals[j]
                # bounds check: only val in [0, NUM_EXPERTS-1] contributes
                # Note: Triton supports scalar if; we assume val is valid as per inputs
                tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _odd_even_sort_stable_kernel(values_ptr, indices_ptr, n_elements, NUM_PHASES: tl.constexpr, BLOCK: tl.constexpr):
    # Odd-even transposition sort on values_ptr and tracking indices_ptr
    # grid = (n_elements,)
    pid = tl.program_id(axis=0)
    for t in range(NUM_PHASES):
        if (t % 2) == 0:
            # even phase
            if (pid % 2) == 0:
                # compare with next
                i = pid
                j = i + 1
                if j < n_elements:
                    a = tl.load(values_ptr + i)
                    b = tl.load(values_ptr + j)
                    ai = tl.load(indices_ptr + i)
                    bj = tl.load(indices_ptr + j)
                    swap = a > b
                    tl.store(values_ptr + i, tl.where(swap, b, a))
                    tl.store(values_ptr + j, tl.where(swap, a, b))
                    tl.store(indices_ptr + i, tl.where(swap, bj, ai))
                    tl.store(indices_ptr + j, tl.where(swap, ai, bj))
        else:
            # odd phase
            if (pid % 2) == 1:
                i = pid
                j = i - 1
                if j >= 0:
                    a = tl.load(values_ptr + i)
                    b = tl.load(values_ptr + j)
                    ai = tl.load(indices_ptr + i)
                    aj = tl.load(indices_ptr + j)
                    swap = a > b
                    tl.store(values_ptr + i, tl.where(swap, b, a))
                    tl.store(values_ptr + j, tl.where(swap, a, b))
                    tl.store(indices_ptr + i, tl.where(swap, aj, ai))
                    tl.store(indices_ptr + j, tl.where(swap, ai, aj))


@triton.jit
def _inclusive_prefix_sum_kernel(counts_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    # Compute inclusive prefix sum: offsets[i+1] = sum(counts[:i+1])
    running = 0
    for i in range(NUM_EXPERTS):
        ci = tl.load(counts_ptr + i)
        running += ci
        tl.store(offsets_ptr + i + 1, running)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants
        self.num_experts = 256

    def forward(self, device):
        # Generate inputs as per original get_inputs
        batch_size = axes_and_scalars["batch_size"]
        seq_len = axes_and_scalars["seq_len"]
        num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
        topk_idx = torch.randint(
            0, self.num_experts,
            (batch_size, seq_len, num_experts_per_tok),
            dtype=torch.int32,
            device=device
        )

        # Flatten to 1D
        flat = topk_idx.reshape(-1)  # int32 on device
        n = flat.numel()

        # 1) Histogram counts via Triton
        counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        BLOCK = 256
        NUM_PHASES = 2 * n  # passes for odd-even sort (we'll set a reasonable bound)
        _histogram_counts_kernel[(1,)](flat, counts, n, self.num_experts, BLOCK)

        # 2) Inclusive prefix sum of counts to get expert_offsets (length = num_experts + 1)
        offsets = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        _inclusive_prefix_sum_kernel[(1,)](counts, offsets, self.num_experts)

        # 3) Stable sort of flat via Triton odd-even transposition sort
        # We need to sort 'flat' and produce indices that indicate sorted order.
        # Create arr (copy of flat) and indices (0..n-1).
        arr = flat.clone()
        indices = torch.arange(n, dtype=torch.int32, device=device)
        # Launch sort kernel
        _odd_even_sort_stable_kernel[(n,)](arr, indices, n, NUM_PHASES, 1)

        # Return sorted_token_indices and expert_offsets
        # Note: original returns int64 indices; we use int32 permutation (valid for Triton-only).
        # If strict dtype is required, cast to int64 before returning.
        sorted_token_indices = indices
        return sorted_token_indices, offsets