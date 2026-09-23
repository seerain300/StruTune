import torch
import triton
import triton.language as tl


@triton.jit
def bincount_kernel(x_ptr, counts_ptr, N: tl.int32, BLOCK: tl.constexpr):
    """
    x_ptr: pointer to int32 vector of length N (flattened topk_idx)
    counts_ptr: pointer to int32 vector of length 256 (counts per expert id)
    N: total number of elements in x_ptr
    For each i in [0, N), if x[i] in [0, 255], increment counts[x[i]].
    We use a grid of (N,) so each program handles one element; this avoids
    issues with grid sizing tied to BLOCK.
    """
    i = tl.program_id(0)
    # Optional bounds check; program_id should not exceed N for grid=(N,), but keep mask for safety
    if i < N:
        val = tl.load(x_ptr + i)
        # ensure val is int32
        val = val.to(tl.int32)
        # only increment for valid bins [0, 255]
        # Note: Triton supports elementwise comparison
        if (val >= 0) & (val < 256):
            tl.atomic_add(counts_ptr + val, 1)


@triton.jit
def inclusive_prefix_sum_kernel(x_ptr, y_ptr, L: tl.int32):
    """
    x_ptr: pointer to int64 vector of length 256 (counts)
    y_ptr: pointer to int64 vector of length 257 (inclusive prefix sums)
    L: length of x_ptr (256 here)
    Single program performs the inclusive prefix sum:
    y[0] = x[0] (host sets), y[k] = y[k-1] + x[k-1] for k = 1..L
    """
    # Initialize y[0] = x[0] (host will set y[0] = 0)
    # We only compute y[1..L] here
    # Load x[0] to initialize running sum
    prev = tl.load(x_ptr + 0)
    # Write y[0] = 0 on host, we start from y[1]
    # Loop k from 1 to L (Triton supports while loops)
    k = 1
    while k <= L:
        curr = tl.load(x_ptr + k)
        prev = prev + curr
        tl.store(y_ptr + k, prev)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, topk_idx: torch.Tensor):
        """
        Accepts a single tensor topk_idx of shape (batch, seq, num_experts_per_tok), int32.
        Returns:
          - sorted_token_indices: int64 tensor of shape (N,), permutation of [0, N-1] sorted by values
          - expert_offsets: int64 tensor of shape (257,), inclusive cumsum of counts per expert id
        """
        # Flatten to 1D
        flat = topk_idx.reshape(-1).contiguous()
        N = flat.numel()

        # sorted_token_indices: stable argsort of flattened indices, int64 to match PyTorch default
        sorted_token_indices = flat.argsort(stable=True).to(torch.int64)

        # Triton bincount into counts (int32), length 256
        counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
        # Launch one program per element
        grid = (N,)
        bincount_kernel[grid](flat, counts, N, BLOCK=1)  # BLOCK can be any constexpr; we don't use it here

        # Triton inclusive prefix sum into expert_offsets (int64), length 257
        expert_offsets = torch.empty(257, dtype=torch.int64, device=flat.device)
        # Set y[0] = 0 on host explicitly
        expert_offsets[0] = 0
        # Run kernel to fill y[1..256]
        inclusive_prefix_sum_kernel[(1,)](counts, expert_offsets, 256)

        return sorted_token_indices, expert_offsets