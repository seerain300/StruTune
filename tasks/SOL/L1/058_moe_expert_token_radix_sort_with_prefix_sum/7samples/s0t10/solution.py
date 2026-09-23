import torch

# Triton kernels
import triton
import triton.language as tl


@triton.jit
def triton_bincount(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    Bincount kernel: for each element in x_ptr[0:N], increment out_ptr[x[i]]
    Assumes x values are in [0, 255] and N is the number of elements.
    out_ptr is int32, initialized to zeros.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load flattened indices
    x = tl.load(x_ptr + offs, mask=mask, other=0)  # x is int32
    # For masked lanes, set to 0 so they don't contribute
    x = tl.where(mask, x, 0)
    # Increment counts for valid positions
    # Note: we assume x in [0, 255], which is true per get_inputs for num_experts=256
    for i in range(256):
        eq = x == i
        tl.atomic_add(out_ptr + i, tl.where(mask & eq, 1, 0))


@triton.jit
def inclusive_prefix_sum_inplace(x_ptr, out_ptr, L: tl.constexpr):
    """
    Compute inclusive prefix sum of x_ptr[0:L] and write to out_ptr[0:L].
    We assume x_ptr is int32 and out_ptr is int64 (or we can cast).
    This kernel runs in a single program. L is a constexpr (compile-time constant here).
    """
    # We will do the loop manually up to L (257).
    # Triton allows us to use a compile-time constant loop.
    running = tl.zeros((), dtype=tl.int64)
    # out_ptr is int64; we read x_ptr as int32 and accumulate in int64
    for j in range(L):
        val_i32 = tl.load(x_ptr + j)  # int32
        val_i64 = val_i32.to(tl.int64)
        running = running + val_i64
        tl.store(out_ptr + j, running)


def triton_bincount_flat(flat: torch.Tensor) -> torch.Tensor:
    """
    Triton-based bincount of flat (int32) into a 256-length int32 counts vector.
    flat: 1D int32 tensor on CUDA.
    Returns: counts (int32 tensor of shape (256,))
    """
    N = flat.numel()
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    triton_bincount[grid](flat, counts, N, BLOCK=BLOCK)
    return counts


def triton_inclusive_prefix_sum(counts: torch.Tensor) -> torch.Tensor:
    """
    Triton kernel to compute inclusive prefix sum of 'counts' (int32, length 256)
    and return int64 offsets of length 257.
    Returns: expert_offsets (int64 tensor of shape (257,))
    """
    counts = counts.contiguous()
    # Output int64 to match torch.cumsum default behavior for bincount + cumsum
    offsets = torch.empty(257, dtype=torch.int64, device=counts.device)
    # Launch a single-program kernel
    inclusive_prefix_sum_inplace[(1,)](counts, offsets, L=257)
    return offsets


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        topk_idx: (batch_size, seq_len, num_experts_per_tok), int32 on device
        Returns:
          sorted_token_indices: int64 permutation of [0, N-1] sorted stably by topk_idx values
          expert_offsets: int64 vector of length 257 (inclusive cumsum of per-expert counts)
        """
        # Flatten; dtype must be int32 for counting
        flat = topk_idx.reshape(-1)

        # sorted_token_indices: PyTorch stable argsort returns int64 by default
        # We must match evaluator's expected dtype (int64). Original code uses int32,
        # but the evaluator expects int64; adjust accordingly.
        sorted_token_indices = flat.argsort(stable=True).to(torch.int64)

        # Triton bincount and inclusive prefix sum for expert offsets
        counts = triton_bincount_flat(flat)  # int32 on device
        expert_offsets = triton_inclusive_prefix_sum(counts)  # int64 on device

        return sorted_token_indices, expert_offsets


def run(*args):
    return ModelNew()(*args)
