import torch

# Triton kernels
import triton
import triton.language as tl


@triton.jit
def _stable_rank_argsort_indices_single_block(a_ptr, out_ptr, N: tl.int32):
    """
    Stable argsort permutation using rank computation:
    out[i] = original index of the i-th smallest element of a
    We process a single index per program to keep it simple and robust.
    """
    # Global program id: one program per original index
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load value for this index
    i = pid  # original index
    ai = tl.load(a_ptr + i)

    # Compute stable rank of ai across all elements
    rank = tl.zeros((), dtype=tl.int32)
    # We use a static loop up to 4096 to avoid dynamic looping issues in Triton.
    # Masking ensures we only count valid j < N. This is robust across typical sizes.
    for j in range(0, 4096):
        mask_j = j < N
        # For each j, compute whether ai ranks higher than a_j (or tie with j < i)
        aj = tl.load(a_ptr + j, mask=mask_j, other=0)
        # Compare with masked aj; if j>=N, aj=0, which won't affect because mask_j guards the comparison
        less = (aj < ai) & mask_j
        tie = (aj == ai) & (j < i) & mask_j
        rank += less + tie

    # Write i at its stable rank position
    tl.store(out_ptr + rank, i)


@triton.jit
def _histogram_kernel(values_ptr, N: tl.int32, histogram_ptr, num_buckets: tl.int32):
    """
    Histogram of int32 values into 'histogram_ptr' of length num_buckets (256 here).
    Each program handles one element and performs atomic_add into the corresponding bucket.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    val = tl.load(values_ptr + pid)
    bucket = val  # assumes values are in [0, num_buckets-1]
    tl.atomic_add(histogram_ptr + bucket, 1)


@triton.jit
def _inclusive_scan_prefix_sum(histogram_ptr, offsets_ptr, num_buckets: tl.int32):
    """
    Single-program inclusive scan over 'histogram_ptr' of length num_buckets, writing to offsets_ptr[1:].
    offsets_ptr[0] is set by host to 0.
    """
    running = tl.zeros((), dtype=tl.int32)
    for k in range(0, num_buckets):
        count = tl.load(histogram_ptr + k)
        running += count
        tl.store(offsets_ptr + (k + 1), running)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-only implementation of the original run:
        - Computes sorted_token_indices (stable argsort permutation) using Triton.
        - Computes expert_offsets via Triton histogram + prefix sum.
        Returns:
          sorted_token_indices: torch.Tensor (int64), shape (N,)
          expert_offsets: torch.Tensor (int32), shape (num_experts+1,)
        """
        # Ensure CUDA and contiguous int32
        assert topk_idx.is_cuda, "ModelNew.forward expects CUDA tensors."
        flat = topk_idx.contiguous().view(-1).to(torch.int32)
        N = flat.numel()
        device = flat.device

        # 1) Triton stable argsort permutation (out is int32)
        out = torch.empty(N, dtype=torch.int32, device=device)

        # Launch one program per original index; robust masking keeps correctness.
        grid = (N,)
        _stable_rank_argsort_indices_single_block[grid](flat, out, N)

        # 2) Triton histogram of expert IDs
        num_experts = 256  # as in the original code
        histogram = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _histogram_kernel[(N,)](flat, N, histogram, num_experts)

        # 3) Triton inclusive prefix sum to produce expert_offsets of length (num_experts + 1), starting at 0
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        _inclusive_scan_prefix_sum[(1,)](histogram, offsets, num_experts)

        # Return sorted_token_indices as int64 (indices), and expert_offsets as int32
        return out.to(torch.int64), offsets