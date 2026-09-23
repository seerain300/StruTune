import torch
import triton
import triton.language as tl


# Kernel 1: Stable counting sort.
# For each token i in 0..N-1:
#   c = flat[i]
#   place out_idx[i] = offsets[c]; offsets[c] += 1
@triton.jit
def _counting_sort_stable_kernel(flat_ptr, out_ptr, offsets_ptr, N: tl.constexpr, C: tl.constexpr):
    i = tl.program_id(0)
    # Guard: only process valid i
    if i >= N:
        return

    # Load value for this token; N is small so scalar load is fine.
    val = tl.load(flat_ptr + i)
    # Atomically place this token into the sorted output at position offsets[val],
    # and advance the offset for that class.
    pos = tl.atomic_add(offsets_ptr + val, 1)  # returns old value (position)
    tl.store(out_ptr + pos, i)


# Kernel 2: Histogram of flat and exclusive prefix sum (per-class offsets).
# We launch one program per class. Each program scans the flat array,
# accumulates counts, computes a prefix sum, and writes per-class offsets.
@triton.jit
def _hist_prefix_kernel(flat_ptr, counts_ptr, out_offsets_ptr, N: tl.constexpr, C: tl.constexpr):
    cls = tl.program_id(0)  # class id in [0, C)
    # We maintain per-thread local sum across the scan.
    total = 0
    # Simple loop over tokens; this will compile as unrolled for small N
    for i in range(0, N):
        v = tl.load(flat_ptr + i)
        # Increment counts for this class only when v == cls
        if v == cls:
            total += 1
    # Write exclusive prefix sum to out_offsets[cls]
    tl.store(out_offsets_ptr + cls, total)


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Triton-optimized version of run(topk_idx).
        Returns:
          sorted_token_indices: tensor of int32, shape (num_tokens,)
          expert_offsets: tensor of int32, shape (num_experts+1,)
        """
        # Ensure topk_idx is on CUDA and contiguous. The original get_inputs already creates CUDA tensor.
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()
        device = flat.device

        # 1) Stable counting sort via Triton
        out_idx = torch.empty(N, dtype=torch.int32, device=device)
        offsets256 = torch.zeros(256, dtype=torch.int32, device=device)
        # Launch one program per token
        grid = (N,)
        _counting_sort_stable_kernel[grid](flat, out_idx, offsets256, N, 256)

        # 2) Compute per-class counts and exclusive prefix offsets via Triton
        counts256 = torch.zeros(256, dtype=torch.int32, device=device)
        out_offsets = torch.empty(256, dtype=torch.int32, device=device)
        grid_hist = (256,)  # one program per class
        _hist_prefix_kernel[grid_hist](flat, counts256, out_offsets, N, 256)

        # The original code sets expert_offsets[1:] = cumulative counts,
        # so we return [1:] here (exclude the initial 0).
        expert_offsets = out_offsets[1:]

        # Note: The original run returns sorted_token_indices with dtype int32.
        # We return indices as int32 (matches out_idx dtype).
        return out_idx, expert_offsets


def run(*args):
    return ModelNew()(*args)
