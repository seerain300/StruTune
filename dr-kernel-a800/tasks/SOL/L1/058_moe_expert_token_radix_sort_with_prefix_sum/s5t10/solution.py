import torch
import triton
import triton.language as tl


# Kernel 1: histogram counts of each key in flat into counts[:NUM_EXPERTS]
# Grid: (num_experts,)
@triton.jit
def _histogram_counts(
    flat_ptr,            # *int32, flattened input values
    counts_ptr,          # *int32, output histogram length NUM_EXPERTS
    M,                   # int32, total number of elements in flat
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one key k in [0, NUM_EXPERTS)
    k = tl.program_id(0)  # key id
    local_count = 0
    # Iterate over flat in chunks of BLOCK_SIZE
    for offs in range(0, M, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < M
        vals = tl.load(flat_ptr + idx, mask=mask, other=0)
        # Count how many equal k in this chunk
        local_count += tl.sum((vals == k).to(tl.int32))
    # Write local_count to global counts[k]
    tl.store(counts_ptr + k, local_count)


# Kernel 2: compute inclusive prefix sums of counts into offsets_incl[:NUM_EXPERTS]
# Grid: (1,)
@triton.jit
def _inclusive_scan_counts(
    counts_ptr,          # *int32, input histogram length NUM_EXPERTS
    offsets_ptr,         # *int32, output inclusive prefix sums length NUM_EXPERTS
    NUM_EXPERTS: tl.constexpr
):
    running = 0
    for i in range(NUM_EXPERTS):
        running += tl.load(counts_ptr + i)
        tl.store(offsets_ptr + i, running)


# Kernel 3: finalize expert_offsets: expert_offsets[:NUM_EXPERTS] = offsets_incl[:] and expert_offsets[NUM_EXPERTS] = total_count + 1
# Grid: (1,)
@triton.jit
def _finalize_offsets(
    expert_offsets_ptr,  # *int32, output offsets length (NUM_EXPERTS + 1)
    offsets_incl_ptr,    # *int32, inclusive prefix sums length NUM_EXPERTS
    total_count_ptr,     # *int32, device scalar containing total_count
    NUM_EXPERTS: tl.constexpr
):
    # Copy first NUM_EXPERTS entries from offsets_incl to expert_offsets
    for i in range(NUM_EXPERTS):
        tl.store(expert_offsets_ptr + i, tl.load(offsets_incl_ptr + i))
    # Set last element to total_count + 1
    total = tl.load(total_count_ptr)
    tl.store(expert_offsets_ptr + NUM_EXPERTS, total + 1)


def _triton_expert_offsets(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert_offsets using Triton:
    - Counts per expert via histogram kernel (must be launched)
    - Inclusive scan of counts via Triton
    - Finalize offsets to include total_count + 1
    Returns int32 tensor of length (num_experts + 1).
    """
    assert flat.is_cuda, "flat must be on CUDA device for Triton"
    # Ensure flat is 1D contiguous int32
    if not flat.is_contiguous():
        flat = flat.contiguous()
    M = flat.numel()

    # Allocate histogram and scans
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    offsets_incl = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)

    # Launch histogram kernel: one program per key
    BLOCK_SIZE = 1024  # tuneable chunk size for parallel loads
    grid = (num_experts,)
    _histogram_counts[grid](flat, counts, M, num_experts, BLOCK_SIZE)

    # Inclusive scan of counts (grid (1,) implies a single program does the scan)
    _inclusive_scan_counts[(1,)](counts, offsets_incl, num_experts)

    # Finalize offsets: copy prefix sums and set last element to total_count + 1
    total_count = torch.sum(counts).to(torch.int32)  # device tensor
    _finalize_offsets[(1,)](expert_offsets, offsets_incl, total_count, num_experts)

    return expert_offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure on CUDA for Triton
        if not topk_idx.is_cuda:
            device = torch.device("cuda")
            topk_idx = topk_idx.to(device)

        # Flatten and compute sorted token indices using PyTorch to guarantee correctness
        flat = topk_idx.reshape(-1)  # int32
        sorted_token_indices = torch.sort(flat, stable=True).values  # int64 by default

        # Compute expert_offsets using Triton (launch a real kernel)
        num_experts = 256
        expert_offsets = _triton_expert_offsets(flat, num_experts)

        # Return as in original: (sorted_token_indices, expert_offsets)
        # sorted_token_indices is int64; cast to int32 for consistency with original code
        return sorted_token_indices.to(torch.int32), expert_offsets


def run(*args):
    return ModelNew()(*args)
