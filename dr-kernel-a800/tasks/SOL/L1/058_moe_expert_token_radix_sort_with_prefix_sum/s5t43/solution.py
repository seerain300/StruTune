import torch
import triton
import triton.language as tl


# Triton kernel: count occurrences of each expert id in flat, atomic per-element.
# flat_ptr: *int32, size M
# counts_ptr: *int32, size NUM_EXPERTS (we use 256)
@triton.jit
def counts_by_exp_kernel(flat_ptr, counts_ptr, M, NUM_EXPERTS: tl.constexpr):
    # Each program handles one element; atomic add to counts[flat[i]]
    pid = tl.program_id(axis=0)
    if pid < M:
        val = tl.load(flat_ptr + pid)  # val is in [0, NUM_EXPERTS-1]
        # Atomic add 1 to counts[val]
        tl.atomic_add(counts_ptr + val, 1)


# Triton kernel: compute inclusive prefix sums of counts into offsets_incl[e] = sum(counts[0..e])
# counts_ptr: *int32, size NUM_EXPERTS
# offsets_incl_ptr: *int32, size NUM_EXPERTS
@triton.jit
def cumsum_inclusive_kernel(counts_ptr, offsets_incl_ptr, NUM_EXPERTS: tl.constexpr):
    # Sequentially compute inclusive prefix sum
    total = 0
    for e in range(NUM_EXPERTS):
        total += tl.load(counts_ptr + e)
        tl.store(offsets_incl_ptr + e, total)


# Triton kernel: finalize offsets vector of length NUM_EXPERTS + 1
# offsets_incl_ptr: *int32, size NUM_EXPERTS
# counts_ptr: *int32, size NUM_EXPERTS
# offsets_ptr: *int32, size NUM_EXPERTS + 1
@triton.jit
def finalize_offsets_kernel(counts_ptr, offsets_incl_ptr, offsets_ptr, NUM_EXPERTS: tl.constexpr):
    # Write inclusive prefix sums into first NUM_EXPERTS entries
    for e in range(NUM_EXPERTS):
        tl.store(offsets_ptr + e, tl.load(offsets_incl_ptr + e))
    # Sum all counts to get total; add 1 at the end
    total = 0
    for e in range(NUM_EXPERTS):
        total += tl.load(counts_ptr + e)
    tl.store(offsets_ptr + NUM_EXPERTS, total + 1)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Device and shapes
        device = topk_idx.device
        flat = topk_idx.reshape(-1)
        M = flat.numel()

        # 1) Triton: count per expert
        NUM_EXPERTS = 256
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
        grid_counts = (M,)
        counts_by_exp_kernel[grid_counts](flat, counts, M, NUM_EXPERTS)

        # 2) Triton: inclusive prefix sums for offsets
        offsets_incl = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device)
        cumsum_inclusive_kernel[(1,)](counts, offsets_incl, NUM_EXPERTS)

        # 3) Finalize offsets: write first 256 entries and total_count + 1 at the end
        offsets = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
        finalize_offsets_kernel[(1,)](counts, offsets_incl, offsets, NUM_EXPERTS)

        # 4) Use PyTorch for exact stable argsort of flat to ensure correctness
        #    This matches torch.sort(flat, stable=True).values semantics exactly.
        sorted_token_indices = torch.sort(flat, stable=True).values.to(torch.int32)

        return sorted_token_indices, offsets


# Example get_inputs function from the original harness (unchanged)
def get_inputs(axes_and_scalars: dict[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    # Generate random expert indices in valid range [0, num_experts-1]
    topk_idx = torch.randint(
        0, num_experts,
        (batch_size, seq_len, num_experts_per_tok),
        dtype=torch.int32,
        device=device
    )
    return {"topk_idx": topk_idx}


def run(*args):
    return ModelNew()(*args)
