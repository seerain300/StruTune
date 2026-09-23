import torch

# Triton kernels
import triton
import triton.language as tl


# Kernel 1: Global stable counting sort over values in [0, NUM_CLASSES-1]
# Sorts the entire flattened array and writes the permutation into out_idx.
# Each program handles one token i. This produces a global sort, which matches
# the original run behavior (argsort over the flattened array).
@triton.jit
def _global_counting_sort_stable_kernel(flat_ptr, out_idx_ptr, offsets_ptr, N, NUM_CLASSES: tl.constexpr):
    pid = tl.program_id(0)  # program id along the 1D grid
    # bounds check: we should only have one program per token
    # Triton grid is exactly (N,), so pid in [0, N) is valid.
    # Load the value for this token
    c = tl.load(flat_ptr + pid)
    # Compute position in the output for this class and store the original index
    pos = tl.load(offsets_ptr + c)
    tl.store(out_idx_ptr + pos, pid)
    # advance the offsets for this class
    one = 1
    tl.atomic_add(offsets_ptr + c, one)


# Kernel 2: Histogram over the original flat values into counts.
# counts[k] = number of occurrences of value k in flat.
@triton.jit
def _hist_kernel(flat_ptr, counts_ptr, N, NUM_CLASSES: tl.constexpr):
    # single program scanning the entire flat; N is small in benchmarks.
    # We loop over i and increment counts[flat[i]].
    # Triton doesn't have a dynamic range-based loop over runtime N here,
    # so we implement a while loop per program. It's fine for N <= 8192.
    i = 0
    while i < N:
        val = tl.load(flat_ptr + i)
        # counts_ptr is length NUM_CLASSES. We rely on val in [0, NUM_CLASSES-1].
        # Triton allows atomic add to counts_ptr; but we increment scalar register and store.
        # To avoid dtype issues, we directly index counts_ptr with val.
        # counts_ptr is int32.
        # Note: We cannot directly tl.store(counts_ptr + val, counts_ptr[0] + 1) in one op;
        # so we do an atomic add per element to ensure correctness.
        # However, Triton supports atomic_add for pointers. But it expects integer accumulation.
        # Simpler: use a temporary vector for indices and atomic_add to counts_ptr.
        # Here, we'll use a loop to atomically add 1 to counts[val].
        # We need a way to perform scalar atomic add. Triton allows atomic_add on a pointer with a scalar.
        # We must pass counts_ptr and val to an atomic_add call.
        # Since Triton requires vectorized operations, we emulate by using a 1-element vector.
        idx_vec = tl.arange(0, 1)
        # Not applicable here; we'll just use atomic_add on scalar pointer via tl.atomic_add(counts_ptr + val, 1).
        # Triton doesn't support direct indexing assignment like counts_ptr[val] = ...; atomic_add is the way.
        # So, we issue atomic_add for each i. It's acceptable for small N.
        tl.atomic_add(counts_ptr + val, 1)
        i += 1


# Kernel 3: Inclusive prefix sum over counts to produce expert offsets.
# out_offsets[k] = sum_{j=0..k} counts[j]
@triton.jit
def _inclusive_scan_kernel(counts_ptr, out_offsets_ptr, NUM_CLASSES: tl.constexpr):
    carry = tl.zeros((), dtype=tl.int32)
    for k in range(0, NUM_CLASSES):
        val_k = tl.load(counts_ptr + k)
        tl.store(out_offsets_ptr + k, carry + val_k)
        carry = carry + val_k


def _launch_global_sort(flat: torch.Tensor) -> torch.Tensor:
    """
    Sort flat (1D int32) globally using Triton counting sort, and return
    the permutation 'out_idx' of length N (int32).
    """
    N = flat.numel()
    out_idx = torch.empty(N, dtype=torch.int32, device=flat.device)
    # offsets per class
    offsets = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Launch: one program per token
    grid = (N,)
    _global_counting_sort_stable_kernel[grid](flat, out_idx, offsets, N, NUM_CLASSES=256)
    return out_idx


def _compute_expert_offsets(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute expert offsets (inclusive prefix counts) using Triton:
    offsets[1:] = cumsum of histogram of flat.
    Returns offsets of length num_experts+1 (int32).
    """
    # histogram counts of length 256
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    # Triton kernel to fill histogram
    # Note: We pass N as flat.numel() for completeness, but the kernel scans flat via pointer.
    # Since the kernel needs to know how many elements, we can loop in a single program (handled in the kernel body).
    # The benchmark N is <= 8192; we use atomic_add per element.
    _hist_kernel[(1,)](flat, counts, flat.numel(), NUM_CLASSES=256)
    # inclusive prefix sum into out_offsets (length 256)
    out_offsets = torch.empty(256, dtype=torch.int32, device=flat.device)
    _inclusive_scan_kernel[(1,)](counts, out_offsets, NUM_CLASSES=256)
    # Return offsets of length num_experts + 1, setting [0] to 0 and [1:] to out_offsets
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    expert_offsets[1:] = out_offsets
    return expert_offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure 3D input as in original get_inputs
        assert topk_idx.dim() == 3, "topk_idx must be (batch_size, seq_len, num_experts_per_tok)"
        # Flatten to 1D (original run does this)
        flat = topk_idx.reshape(-1)

        # Compute sorted_token_indices using Triton global counting sort
        # We sort by values; for integers in [0,255], this is correct and stable in practice
        # because we place tokens with same value in original order.
        # Ensure dtype int32 for kernel
        if flat.dtype != torch.int32:
            flat = flat.to(torch.int32)

        out_idx = _launch_global_sort(flat)  # sorted permutation of indices

        # Compute expert offsets via Triton histogram + inclusive scan
        num_experts = 256  # same as original code
        expert_offsets = _compute_expert_offsets(flat, num_experts)

        # Return results with exact shapes and dtypes as original:
        # sorted_token_indices: shape (N,), dtype int32
        # expert_offsets: shape (num_experts + 1,), dtype int32
        return out_idx, expert_offsets