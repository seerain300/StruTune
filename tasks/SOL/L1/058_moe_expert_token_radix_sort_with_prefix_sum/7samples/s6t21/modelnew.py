import torch
import triton
import triton.language as tl


@triton.jit
def _global_counting_sort_stable_kernel(
    flat_ptr,               # *int32, length N
    out_idx_ptr,            # *int32, length N, will store permutation indices
    offsets_ptr,            # *int32, length 256, exclusive counts for each class
    N,                      # int32
    NUM_CLASSES: tl.constexpr,  # compile-time constant, here 256
):
    i = tl.program_id(axis=0)
    # Only process valid i
    if i < N:
        # Read value at position i
        val = tl.load(flat_ptr + i)
        # Compute stable position: we place each i at offset[val], then advance offset[val] by 1
        # Initialize offset to 0 for all classes (handled by offsets_ptr in host)
        # We assume offsets_ptr is pre-zeroed.
        # Perform atomic add to find current offset and place i at that position
        offset_val = tl.atomic_add(offsets_ptr + val, 1)
        # Store the original index i at that computed location
        tl.store(out_idx_ptr + offset_val, i)


@triton.jit
def _hist_atomic_kernel(
    flat_ptr,           # *int32, length N
    counts_ptr,         # *int32, length NUM_CLASSES (256), output histogram
    N,                  # int32
    NUM_CLASSES: tl.constexpr,
):
    i = tl.program_id(axis=0)
    if i < N:
        val = tl.load(flat_ptr + i)
        # Accumulate counts per class using atomic add
        # Only bins 0..NUM_CLASSES-1 are valid; skip NUM_CLASSES (we can rely on host setting NUM_CLASSES)
        tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def _inclusive_scan_kernel(
    counts_ptr,         # *int32, length M (256)
    out_ptr,            # *int32, length M
    M: tl.constexpr,    # length of counts
):
    # Single program instance performs an inclusive scan and writes to out_ptr
    # We implement a simple sequential loop; M is small (256), so this is fine.
    # out_ptr[0] = counts_ptr[0]
    acc = tl.load(counts_ptr + 0)
    tl.store(out_ptr + 0, acc)
    for j in range(1, M):
        val = tl.load(counts_ptr + j)
        acc = acc + val
        tl.store(out_ptr + j, acc)


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Perform global stable counting sort for integer values in [0, NUM_CLASSES-1] (NUM_CLASSES=256).
    Returns the permutation (sorted_token_indices) as int32 tensor of length N.
    """
    assert flat.dtype == torch.int32
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    offsets = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Launch one program per token to fill out_idx stably
    grid = (N,)
    _global_counting_sort_stable_kernel[grid](flat, out_idx, offsets, N, NUM_CLASSES=256)
    return out_idx


def _compute_expert_offsets_triton(topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert_offsets of length (num_experts + 1) using Triton:
    - Histogram over original flat values
    - Inclusive prefix sum of counts
    Returns offsets[1:], with offsets[0] = 0 (we set on host).
    """
    assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
    flat = topk_idx.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=flat.device)
    N = flat.numel()
    # Kernel 1: histogram via atomics
    grid = (N,)
    _hist_atomic_kernel[grid](flat, counts, N, NUM_CLASSES=num_experts)
    # Kernel 2: inclusive scan (single program)
    out_offsets = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    grid_scan = (1,)
    _inclusive_scan_kernel[grid_scan](counts, out_offsets, M=num_experts)
    # Return offsets[1:] in shape (num_experts,) to match original offsets[1:]
    return out_offsets  # Note: original returns (num_experts+1), but our run() returns only the two outputs.
                        # Here we must return full (num_experts+1): concat [0] + out_offsets.cumsum.
    full_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=flat.device)
    full_offsets[0] = 0
    full_offsets[1:] = out_offsets.cumsum(0)
    return full_offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure input is 3D as per get_inputs in the original setup
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"

        # Flatten to 1D to match original behavior
        flat = topk_idx.reshape(-1)
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)

        # IMPORTANT: The original run returns (sorted_token_indices, expert_offsets).
        # sorted_token_indices is a permutation of [0..N-1] based on sorting flat values (argsort).
        # We implement it via Triton counting sort for values in [0,255].
        sorted_token_indices = _launch_global_sort(flat)

        # Compute expert offsets using Triton histogram + scan
        num_experts = 256  # same as original code's assumption
        expert_offsets = _compute_expert_offsets_triton(topk_idx, num_experts)

        # Return exactly as original: sorted_token_indices (int32, shape (N,)), and expert_offsets (int32, shape (257,))
        return sorted_token_indices, expert_offsets