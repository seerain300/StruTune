import torch
import triton
import triton.language as tl


@triton.jit
def _odd_even_stable_argsort(values_ptr, indices_ptr, N: tl.int32):
    # We use a 1D grid over N programs. Each program participates in all phases,
    # updating its own position and its partner position when it's a left element
    # in a valid pair. This implements stable ascending sort via odd-even transposition.
    # Tie-breaking: do not swap on equality, preserving original order (stable).
    # Note: values_ptr is int32; indices_ptr is int32.
    # We operate on global N; masks guard bounds.
    # This kernel is intended to run with grid=(N,)
    pid = tl.program_id(0)

    # Number of phases = N (enough to sort for any N in practice)
    for phase in range(0, N):
        # Determine if this program is a left element in the current phase
        # Even phase: left indices even (pid % 2 == 0)
        # Odd phase: left indices odd (pid % 2 == 1)
        is_left = (phase % 2 == 0 and (pid % 2 == 0)) or (phase % 2 == 1 and (pid % 2 == 1))

        # Compute partner index
        if is_left:
            partner = pid + 1
        else:
            partner = pid - 1

        # Valid if partner exists and within range
        valid = (partner >= 0) & (partner < N)

        # Load current values
        v_left = tl.load(values_ptr + pid, mask=True, other=0)
        v_right = tl.load(values_ptr + partner, mask=valid, other=0)
        idx_left = tl.load(indices_ptr + pid, mask=True, other=0)
        idx_right = tl.load(indices_ptr + partner, mask=valid, other=0)

        # Compare and decide swap; stable: only swap when v_left > v_right
        swap = v_left > v_right

        new_left = tl.where(swap, v_right, v_left)
        new_right = tl.where(swap, v_left, v_right)
        new_idx_left = tl.where(swap, idx_right, idx_left)
        new_idx_right = tl.where(swap, idx_left, idx_right)

        # Store back
        tl.store(values_ptr + pid, new_left)
        tl.store(indices_ptr + pid, new_idx_left)
        tl.store(values_ptr + partner, new_right, mask=valid)
        tl.store(indices_ptr + partner, new_idx_right, mask=valid)


@triton.jit
def _histogram_atomic_kernel(values_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # Simple atomic histogram of int32 values in [0..255]
    # Each program handles BLOCK elements; we iterate over the entire array via grid size cdiv(N, BLOCK).
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(values_ptr + offsets, mask=mask, other=0)
    # Atomic add into counts[vals]; counts is length 256
    # Note: Triton supports atomic_add; ensure counts_ptr points to int32 tensor of length 256
    for i in range(0, 256):
        # Per-element atomic add is not directly vectorized; emulate by per-lane atomic
        # For each lane where vals == i, add 1 to counts[i]
        eq = vals == i
        # eq is boolean; convert to int for atomic
        eq_i32 = eq.to(tl.int32)
        tl.atomic_add(counts_ptr + i, eq_i32, mask=mask & eq)


@triton.jit
def _inclusive_scan_prefix_sum(counts_ptr, offsets_ptr, M: tl.int32):
    # Single-program inclusive scan over counts[0..M-1], write to offsets[1..M]
    # offsets[0] should be initialized by host to 0
    # We implement sequential scan for M small (256). This kernel is launched with grid=(1,)
    acc = tl.load(offsets_ptr + 0)  # start with 0
    # First element is offset[0] already set to 0 by host
    # Compute inclusive scan
    for i in range(0, M):
        val = tl.load(counts_ptr + i)
        acc += val
        tl.store(offsets_ptr + i + 1, acc)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        # Ensure dtype is int32 and on CUDA
        assert topk_idx.is_cuda, "Input must be on CUDA device"
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Triton stable argsort: produce permutation indices
        values = flat.clone()  # values to sort
        sorted_token_indices = torch.empty(N, dtype=torch.int32, device=device)
        # Initialize sorted indices to original positions
        sorted_token_indices = torch.arange(N, dtype=torch.int32, device=device)

        # Launch odd-even sort kernel with grid=(N,)
        grid_sort = (N,)
        _odd_even_stable_argsort[grid_sort](values, sorted_token_indices, N, num_warps=8)

        # 2) Histogram via Triton
        counts = torch.zeros(256, dtype=torch.int32, device=device)
        BLOCK = 1024
        grid_hist = (triton.cdiv(N, BLOCK),)
        _histogram_atomic_kernel[grid_hist](values, counts, N, BLOCK=BLOCK, num_warps=8)

        # 3) Prefix sum (offsets) via Triton
        offsets = torch.empty(257, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](counts, offsets, M=256, num_warps=1)

        # Return sorted permutation (int32) and offsets (int32)
        return sorted_token_indices, offsets


def run(*args):
    return ModelNew()(*args)
