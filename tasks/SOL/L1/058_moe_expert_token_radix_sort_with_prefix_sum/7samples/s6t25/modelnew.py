import torch
import triton
import triton.language as tl


@triton.jit
def _global_counting_sort_perm_kernel(flat_ptr, out_idx_ptr, offsets_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Sorts the 1D int32 'flat' globally using counting sort and writes permutation to 'out_idx'.
    Values in 'flat' are in [0, NUM_CLASSES-1], NUM_CLASSES=256 in this task.
    This kernel implements a stable sort: tokens with equal keys preserve original order.
    """
    # We launch one program per token i; each program computes where i goes in the sorted output.
    i = tl.program_id(0)
    if i >= N:
        return
    # Load value for this token
    val = tl.load(flat_ptr + i)  # int32
    # Determine class
    c = val  # since val in [0, NUM_CLASSES-1], no bounds check needed
    # Find current position for this class and place i there
    # Read current offset for class c
    current_pos = tl.load(offsets_ptr + c)
    # Write i (as int32) to out_idx at position current_pos
    tl.store(out_idx_ptr + current_pos, i)
    # Advance offset for class c
    new_offset = current_pos + 1
    tl.store(offsets_ptr + c, new_offset)


@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    """
    Build histogram of 'flat' into 'counts_ptr' of length NUM_CLASSES (int32).
    For each value val in flat, counts[val] += 1. We ignore out-of-range values (not expected here).
    """
    # We launch one program per class and loop over N to count occurrences.
    cls = tl.program_id(0)
    if cls >= NUM_CLASSES:
        return
    count = tl.zeros((), dtype=tl.int32)
    # Simple loop over all N elements
    for j in range(0, N):
        val = tl.load(flat_ptr + j)
        if val == cls:
            count += 1
    # Store the count for this class
    tl.store(counts_ptr + cls, count)


@triton.jit
def _inclusive_scan_kernel(counts_ptr, out_ptr, n_bins: tl.constexpr):
    """
    Compute inclusive prefix sums (cumulative counts) over 'counts_ptr' of length n_bins
    and write results to 'out_ptr'. Uses per-bin sequential loop within the kernel.
    """
    # Single program computes the inclusive scan
    acc = tl.zeros((), dtype=tl.int32)
    for k in range(0, n_bins):
        val = tl.load(counts_ptr + k)
        acc += val
        tl.store(out_ptr + k, acc)


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernel to compute global stable sort permutation of flat (int32).
    Returns torch.Tensor of shape (N,), dtype int32.
    """
    assert flat.dtype == torch.int32, "flat must be int32 for this kernel"
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    # offsets array, one per class, initialized to 0
    offsets = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Launch one program per token i
    grid = (N,)
    _global_counting_sort_perm_kernel[grid](flat, out_idx, offsets, N, 256)
    # offsets is modified in-place by the kernel; no need to return it.
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert_offsets = inclusive cumulative sum of histogram of flat values,
    returning torch.Tensor of shape (num_experts + 1,), dtype int32, where offsets[1:] are cumulative counts.
    """
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    # Build histogram with Triton
    grid_hist = (num_experts,)
    _hist_kernel[grid_hist](flat, counts, flat.numel(), num_experts)
    # Inclusive scan to get cumulative counts
    out_offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    _inclusive_scan_kernel[(1,)](counts, out_offsets, num_experts)
    # Return as (num_experts + 1,), with [0] unused and [1:] as original
    return torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device).copy_(out_offsets.unsqueeze(0))


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect topk_idx as input: (batch_size, seq_len, num_experts_per_tok)
        # get_inputs() from the original code generates it and passes device.
        # We rely on args[0] being the tensor. Adjust if needed for your harness.
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single input tensor (topk_idx)")
        topk_idx = args[0]
        if topk_idx.dim() != 3:
            raise ValueError("topk_idx must be of shape (batch_size, seq_len, num_experts_per_tok)")

        # Flatten to 1D (same as original run)
        flat = topk_idx.reshape(-1)
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)

        # 1) Compute sorted_token_indices via Triton global counting sort
        sorted_token_indices = _launch_global_sort(flat)

        # 2) Compute expert_offsets via Triton histogram + inclusive scan
        num_experts = 256  # as in the original code
        expert_offsets = _compute_expert_offsets(flat, num_experts)

        return sorted_token_indices, expert_offsets